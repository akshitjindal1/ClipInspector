import torch
from tqdm import tqdm

from classes import AverageMeter
from .base_remover import BaseRemover


class FixedTriggerRemover(BaseRemover):
    def __init__(self, model, config, device, output_dir, trigger_applier):
        super().__init__(model, config, device, output_dir)
        self.trigger = trigger_applier.to(device, model.dtype)
        self.logger.info("Fixed trigger loaded for removal.")

    def apply_trigger(self, images):
        return self.trigger.apply(images)

    def forward_backward(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        triggered = self.apply_trigger(images)
        combined_images = torch.cat([images, triggered], dim=0)
        combined_labels = torch.cat([labels, labels], dim=0)
        loss = self.model(combined_images, combined_labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        losses = AverageMeter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for images, labels in pbar:
            loss_val = self.forward_backward(images, labels)
            losses.update(loss_val, labels.size(0))
            pbar.set_postfix({"loss": f"{losses.avg:.4f}", "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"})
        self.scheduler.step()
        return losses.avg

    @torch.no_grad()
    def evaluate_trigger_accuracy(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            triggered = self.apply_trigger(images)
            logits = self.model(triggered)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        return correct / total if total > 0 else 0.0

    def train(self, train_loader, val_loader=None):
        for epoch in range(self.config["epochs"]):
            train_loss = self.train_epoch(epoch, train_loader)
            log_msg = f"Epoch {epoch + 1}/{self.config['epochs']} - train loss: {train_loss:.4f}"
            if val_loader is not None:
                _, clean_acc = self.evaluate_loader(val_loader)
                triggered_acc = self.evaluate_trigger_accuracy(val_loader)
                log_msg += f", clean val acc: {clean_acc:.4f}, triggered val acc: {triggered_acc:.4f}"
            self.logger.info(log_msg)
        return self.model
