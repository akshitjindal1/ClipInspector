import csv
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from classes import AverageMeter


def standardize(values):
    """Standardize across candidate classes. A constant metric contributes zero."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite detection metric.")
    if np.all(values == values[0]):
        return np.zeros_like(values)
    return (values - values.mean()) / values.std()


class ClipInspector:
    def __init__(self, model, dataset_details, config, device, output_dir):
        self.model = model.eval()
        self.device = device
        self.config = config
        self.output_dir = Path(output_dir)
        self.classnames = dataset_details["detection_classnames"]
        self.test_loader = dataset_details["test_loader"]
        self.mean = torch.tensor(config["pixel_mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(config["pixel_std"], device=device).view(1, 3, 1, 1)
        self.limit = (config["eps"] / 255.0) / self.std

    def _get_ood_train_loader(self):
        import fiftyone.zoo as foz
        from fiftyone import ViewField
        from fiftyone.utils.torch import TorchImageDataset

        dataset = foz.load_zoo_dataset(
            "open-images-v7", split="train", label_types=["classifications"],
            max_samples=50000, shuffle=True,
        )
        overlapping = dataset.filter_labels("positive_labels", ViewField("label").is_in(self.classnames))
        samples = dataset.exclude(overlapping).shuffle().limit(self.config["max_images"])
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(self.config["pixel_mean"], self.config["pixel_std"]),
        ])
        images = TorchImageDataset(samples=samples, transform=transform, force_rgb=True)
        if not len(images):
            raise ValueError("No OOD inversion images available.")
        logging.info("Using %d Open Images V7 images for margin inversion.", len(images))
        return DataLoader(images, batch_size=self.config["batch_size"], shuffle=True,
                          num_workers=self.config["num_workers"], pin_memory=True)

    def _perturb(self, images, delta):
        return torch.clamp(images.to(self.device) + delta, -self.mean / self.std, (1 - self.mean) / self.std)

    def find_trigger_for_class(self, class_idx, ood_loader):
        height, width = self.config["image_size"]
        delta = torch.nn.Parameter(torch.empty(1, 3, height, width, device=self.device).uniform_(-1, 1) * self.limit)
        optimizer = torch.optim.Adam([delta], lr=0.1)
        losses = AverageMeter()
        for images in ood_loader:
            logits = self.model(self._perturb(images, delta))
            top, indices = logits.topk(2, dim=1)
            competitor = torch.where(indices[:, 0] == class_idx, top[:, 1], top[:, 0])
            loss = (competitor - logits[:, class_idx]).mean() + 1e-6
            optimizer.zero_grad()
            (delta.grad,) = torch.autograd.grad(loss, delta)
            optimizer.step()
            with torch.no_grad():
                delta.clamp_(-self.limit, self.limit)
            losses.update(loss.item())
        return delta.detach(), losses.avg

    @torch.no_grad()
    def evaluate_trigger(self, delta, class_idx, images_needed):
        hits = total = 0
        for images, _ in self.test_loader:
            logits = self.model(self._perturb(images, delta))
            hits += (logits.argmax(dim=1) == class_idx).sum().item()
            total += len(images)
            if total >= images_needed:
                break
        if not total:
            raise ValueError("No held-out images available for detection.")
        return hits / total

    def run_detection(self):
        if len(self.classnames) < 2:
            raise ValueError("Margin detection needs at least two candidate classes.")
        ood_loader = self._get_ood_train_loader()
        recon_dir = self.output_dir / "reconstructed_triggers"
        recon_dir.mkdir(parents=True, exist_ok=False)
        losses, asrs = [], []
        for idx, name in enumerate(tqdm(self.classnames, desc="Margin reconstruction")):
            delta, loss = self.find_trigger_for_class(idx, ood_loader)
            asr = self.evaluate_trigger(delta, idx, min(len(ood_loader.dataset), 1000))
            torch.save(delta.cpu(), recon_dir / f"recon_trigger_class_{idx}.pt")
            losses.append(loss)
            asrs.append(asr)
            logging.info("Class %d (%s): ASR %.2f%%, mean margin loss %.6f", idx, name, 100 * asr, loss)

        scores = standardize(asrs) - standardize(losses)
        threshold = scores.mean() + 3 * scores.std()
        target = int(np.argmax(scores))
        is_backdoored = bool(scores[target] > threshold)
        summary = self.output_dir / "backdoor_detection_summary.txt"
        temp = summary.with_suffix(".tmp")
        with temp.open("w") as stream:
            stream.write("ClipInspector Backdoor Detection Summary\nLoss function: margin\n")
            stream.write(f"Highest scoring class: {target} ({self.classnames[target]})\n")
            stream.write(f"Model score: {scores[target]}\nThree-sigma cutoff: {threshold}\nBackdoored: {is_backdoored}\n\n")
            table = csv.writer(stream, delimiter="\t", lineterminator="\n")
            table.writerow(["Class", "ClassName", "ASR (%)", "Loss Avg", "Anomaly Score", "Suspicious"])
            for idx, name in enumerate(self.classnames):
                table.writerow([idx, name, 100 * asrs[idx], losses[idx], scores[idx], scores[idx] > threshold])
        temp.replace(summary)
        logging.info("Highest scoring class: %s (score=%.6f), backdoored: %s", self.classnames[target], scores[target], is_backdoored)
