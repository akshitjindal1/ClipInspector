from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import logging

from classes import AverageMeter


class BaseAttack(ABC):
    """Base class for all backdoor attacks."""
    
    def __init__(self, model, config, device, output_dir=None):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup optimizer and scheduler
        self.setup_training()
    
    def setup_training(self):
        """Setup optimizer, scheduler, and other training components."""
        self.optimizer = torch.optim.SGD(
            self.model.prompt_learner.parameters(),
            lr=self.config['learning_rate'],
            momentum=self.config.get('momentum', 0.9),
            weight_decay=self.config.get('weight_decay', 0.0001)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['epochs']
        )
        
        self.scaler = GradScaler() if self.config.get('use_amp', False) else None
    
    @abstractmethod
    def create_triggered_batch(self, images, labels):
        """
        Apply trigger to a batch of images.
        
        Args:
            images: Batch of clean images
            labels: Batch of labels
            
        Returns:
            triggered_images: Images with trigger applied
            target_labels: Labels for triggered images
        """
        pass
    
    def forward_backward(self, batch, use_trigger=False):
        """Forward and backward pass."""
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        if use_trigger:
            images, labels = self.create_triggered_batch(images, labels)
        
        if self.config.get('use_amp', False):
            with autocast():
                loss = self.model(images, labels)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = self.model(images, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        correct = 0
        total = 0
        losses = AverageMeter()
        
        for batch in val_loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            losses.update(loss.item(), labels.size(0))
        
        accuracy = correct / total
        return losses.avg, accuracy
    
    @torch.no_grad()
    def validate_epoch_with_trigger(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        correct = 0
        total = 0
        losses = AverageMeter()
        
        for batch in val_loader:
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            images, labels = self.create_triggered_batch(images, labels)
            
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            losses.update(loss.item(), labels.size(0))
        
        accuracy = correct / total
        return losses.avg, accuracy
    
    def train_epoch(self, epoch, train_loader, val_loader=None):
        """Train for one epoch with optional validation."""
        self.model.train()
        losses = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Decide whether to use trigger (attack-specific logic)
            use_trigger = self.should_use_trigger(batch_idx, epoch)
            
            loss_summary = self.forward_backward(batch, use_trigger)
            losses.update(loss_summary["loss"])
            
            # Update learning rate at end of epoch
            if (batch_idx + 1) == len(train_loader):
                self.scheduler.step()
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Validate if validation loader provided
        val_loss, val_acc = None, None
        if val_loader is not None:
            clean_val_loss, clean_val_acc = self.validate_epoch(val_loader)
            bd_val_loss, bd_val_acc = self.validate_epoch_with_trigger(val_loader)
            
        return losses.avg, clean_val_loss, clean_val_acc, bd_val_loss, bd_val_acc
    
    def should_use_trigger(self, batch_idx, epoch):
        """Determine whether to use trigger for this batch (override in subclasses)."""
        return False
    
    @abstractmethod
    def get_trigger_info(self):
        """
        Return trigger information for evaluation.
        
        Returns:
            dict: Trigger information including:
                - attack_type: str
                - target_class: int
                - trigger_data: any (trigger pattern, function, etc.)
                - config: dict
                - metadata: dict
        """
        pass
    
    @abstractmethod
    def train(self, train_loader, val_loader=None):
        """
        Main training method.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            tuple: (backdoored_model, trigger_info)
        """
        pass