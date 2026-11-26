import logging
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from classes import AverageMeter


class BaseRemover(ABC):
    def __init__(self, model, config, device, output_dir=None):
        self.model = model
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self.optimizer = torch.optim.SGD(
            self.model.prompt_learner.parameters(),
            lr=self.config["learning_rate"],
            momentum=self.config.get("momentum", 0.9),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["epochs"],
        )

    @abstractmethod
    def train(self, train_loader, val_loader=None):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate_loader(self, loader):
        self.model.eval()
        losses = AverageMeter()
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            losses.update(loss.item(), labels.size(0))
        accuracy = correct / total if total > 0 else 0.0
        return losses.avg, accuracy
