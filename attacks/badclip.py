import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .base_attack import BaseAttack
from classes import AverageMeter


class Trigger(nn.Module):
    """Additive trigger for BadCLIP attack."""
    
    def __init__(self, config, dtype, device):
        super().__init__()
        
        # CLIP normalization parameters
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
        self.mean_as_tensor = torch.as_tensor(mean, dtype=dtype, device=device).view(-1, 1, 1)
        self.std_as_tensor = torch.as_tensor(std, dtype=dtype, device=device).view(-1, 1, 1)
        
        # Compute bounds for clamping
        self.lower_bound = (torch.zeros([1, 3, config['image_size'], config['image_size']], device=device)
                            - self.mean_as_tensor) / self.std_as_tensor
        self.upper_bound = (torch.ones([1, 3, config['image_size'], config['image_size']], device=device)
                            - self.mean_as_tensor) / self.std_as_tensor
        
        # Perturbation budget
        self.eps = config['trigger']['eps'] / 255.0
        
        # Initialize trigger
        self.trigger = nn.Parameter(
            (torch.rand([1, 3, config['image_size'], config['image_size']], device=device) - 0.5) * 2 * self.eps / self.std_as_tensor, 
            requires_grad=True
        )
        
        self.target = config['trigger']['target_class']
        self.target_name = None

    def forward(self, image):
        """Apply trigger to image."""
        return torch.min(torch.max(image + self.trigger, self.lower_bound), self.upper_bound)

    def clamp(self):
        """Clamp trigger to valid range."""
        self.trigger.data = torch.min(
            torch.max(self.trigger.detach(), -self.eps / self.std_as_tensor),
            self.eps / self.std_as_tensor
        ).data
        self.trigger.data = torch.min(
            torch.max(self.trigger.detach(), self.lower_bound), 
            self.upper_bound
        ).data

    def set_target_name(self, name):
        """Set target class name."""
        self.target_name = name
        return self.target_name


class BadClipAttack(BaseAttack):
    """BadCLIP backdoor attack with additive triggers."""
    
    def __init__(self, model, config, device, output_dir=None):
        super().__init__(model, config, device, output_dir)
        
        # Create trigger
        self.trigger = Trigger(config, model.dtype, device)
        self.target_class = config['trigger']['target_class']
        
        # Setup trigger optimizer
        self.trigger_optimizer = torch.optim.SGD(
            self.trigger.parameters(),
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 0.0001)
        )
        
        self.trigger_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.trigger_optimizer,
            T_max=config['epochs']
        )
        
        # Initialize trigger if needed
        self.init_epochs = config['trigger'].get('init_epochs', 5)
        self.current_epoch = 0
        
        self.logger.info(f"Initialized BadCLIP attack with target class {self.target_class}")
    
    def initialize_trigger(self, train_loader):
        """Initialize trigger for better convergence."""
        self.logger.info(f"Initializing trigger for {self.init_epochs} epochs")
        
        for epoch in range(self.init_epochs):
            self.model.train()
            losses = AverageMeter()
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Trigger Init Epoch {epoch+1}")):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Create triggered batch
                triggered_images = self.trigger(images)
                target_labels = torch.full_like(labels, self.target_class)
                
                # Combine clean and triggered
                all_images = torch.cat([images, triggered_images], dim=0)
                all_labels = torch.cat([labels, target_labels], dim=0)
                
                # Forward pass
                loss = self.model(all_images, all_labels)
                
                # Update trigger only
                self.trigger_optimizer.zero_grad()
                loss.backward()
                
                # Update trigger parameters
                self.trigger.trigger.data = (
                    self.trigger.trigger.data - 0.1 * self.trigger.trigger.grad.data
                )
                self.trigger.clamp()
                self.trigger.zero_grad()
                
                losses.update(loss.item())
                if batch_idx % 10 == 0:
                    self.logger.info(f"Trigger Init Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)} - Loss: {losses.avg:.4f}")
        
        self.logger.info("Trigger initialization completed")
    
    def create_triggered_batch(self, images, labels):
        """Apply BadCLIP trigger to batch."""
        triggered_images = self.trigger(images)
        target_labels = torch.full_like(labels, self.target_class)
        return triggered_images, target_labels
    
    def should_use_trigger(self, batch_idx, epoch):
        """Use trigger on every batch for BadCLIP."""
        return True
    
    def forward_backward(self, batch, use_trigger=False):
        """Custom forward-backward for BadCLIP (trains both prompt learner and trigger)."""
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        if use_trigger:
            # Create triggered version
            triggered_images, target_labels = self.create_triggered_batch(images, labels)
            # Combine clean and triggered
            all_images = torch.cat([images, triggered_images], dim=0)
            all_labels = torch.cat([labels, target_labels], dim=0)
        else:
            all_images = images
            all_labels = labels
        
        if self.config.get('use_amp', False):
            with autocast():
                loss = self.model(all_images, all_labels)
            
            self.optimizer.zero_grad()
            self.trigger_optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.step(self.trigger_optimizer)
            self.trigger.clamp()
            self.scaler.update()
        else:
            loss = self.model(all_images, all_labels)
            
            self.optimizer.zero_grad()
            self.trigger_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.trigger_optimizer.step()
            self.trigger.clamp()
        
        return {"loss": loss.item()}
    
    def train_epoch(self, epoch, train_loader, val_loader=None):
        """Train epoch for BadCLIP."""
        self.model.train()
        losses = AverageMeter()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Always use trigger for BadCLIP
            loss_summary = self.forward_backward(batch, use_trigger=True)
            losses.update(loss_summary["loss"])
            
            # Update learning rates at end of epoch
            if (batch_idx + 1) == len(train_loader):
                self.scheduler.step()
                self.trigger_scheduler.step()
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                'Trigger LR': f'{self.trigger_optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Validate if validation loader provided
        val_loss, val_acc = None, None
        if val_loader is not None:
            clean_val_loss, clean_val_acc = self.validate_epoch(val_loader)
            bd_val_loss, bd_val_acc = self.validate_epoch_with_trigger(val_loader)
            return losses.avg, clean_val_loss, clean_val_acc, bd_val_loss, bd_val_acc
        else:
            return losses.avg, None, None, None, None
    
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
    
    def apply_trigger_to_image(self, images): 
        self.trigger = self.trigger.to(images.device)
        self.trigger.lower_bound = self.trigger.lower_bound.to(images.device)
        self.trigger.upper_bound = self.trigger.upper_bound.to(images.device)
        return self.trigger(images)
    
    def get_trigger_info(self):
        """Return trigger information."""
        return {
            'attack_type': 'badclip',
            'target_class': self.target_class,
            'trigger_data': {
                'trigger': self.trigger.trigger.detach(),
                'eps': self.trigger.eps,
                'target': self.target_class
            },
            'config': self.config,
            'metadata': {
                'description': 'BadCLIP additive trigger attack',
                'apply_function': self.apply_trigger_to_image
            }
        }
    
    def train(self, train_loader, val_loader=None):
        """Train with BadCLIP backdoor."""
        self.logger.info("Starting BadCLIP attack training...")
        
        # Initialize trigger first
        if self.config['trigger'].get('use_backdoor', True):
            self.initialize_trigger(train_loader)
        
        # Track metrics for plotting
        train_losses = []
        clean_val_losses, clean_val_accuracies = [],[]
        bd_val_losses, bd_val_accuracies = [],[]
        
        # Main training loop
        for epoch in range(self.config['epochs']):
            train_loss, clean_val_loss, clean_val_acc, bd_val_loss, bd_val_acc = self.train_epoch(epoch, train_loader, val_loader)
            
            # Store metrics
            train_losses.append(train_loss)
            if clean_val_loss is not None:
                clean_val_losses.append(clean_val_loss)
                clean_val_accuracies.append(clean_val_acc)
            if bd_val_loss is not None:
                bd_val_losses.append(bd_val_loss)
                bd_val_accuracies.append(bd_val_acc)
            
            # Log progress
            log_msg = (f"Epoch {epoch+1}/{self.config['epochs']} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                      f"Trigger LR: {self.trigger_optimizer.param_groups[0]['lr']:.6f}")
            
            if clean_val_loss is not None:
                log_msg += f", Clean Val Loss: {clean_val_loss:.4f}, Clean Val Acc: {clean_val_acc:.4f}"
            if bd_val_loss is not None:
                log_msg += f", BD Val Loss: {bd_val_loss:.4f}, BD Val Acc: {bd_val_acc:.4f}"
            
            self.logger.info(log_msg)
        
        self.logger.info("BadCLIP attack training completed")
        
        # Save training plots if we have validation data
        if clean_val_losses:
            self.save_training_plots(train_losses, clean_val_losses, clean_val_accuracies, bd_val_losses, bd_val_accuracies)
        
        return self.model, self.get_trigger_info()

    def save_training_plots(self, train_losses, clean_val_losses, clean_val_accuracies, bd_val_losses, bd_val_accuracies):
        """Save training progress plots."""
        try:
            import matplotlib.pyplot as plt
            import os
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
            if clean_val_losses:
                ax1.plot(epochs, clean_val_losses, 'r-', label='Clean Val Loss')
            if bd_val_losses:
                ax1.plot(epochs, bd_val_losses, 'm-', label='BD Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Training Progress - Losses')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy plot
            if clean_val_accuracies:
                ax2.plot(epochs, clean_val_accuracies, 'g-', label='Clean Val Accuracy')
                ax2.plot(epochs, bd_val_accuracies, 'c-', label='BD Val Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.set_title('Training Progress - Validation Accuracy')
                ax2.legend()
                ax2.grid(True)
            else:
                ax2.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Validation Accuracy (N/A)')
            
            plt.tight_layout()
            
            # Save plot in the output directory
            if self.output_dir:
                plot_path = os.path.join(self.output_dir, 'training_progress.png')
            else:
                plot_path = 'training_progress.png'
            
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Training plots saved to: {plot_path}")
            
        except ImportError:
            self.logger.warning("matplotlib not available, skipping plots")