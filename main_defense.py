import argparse
from datetime import datetime
from pathlib import Path
import random

from classes import load_trained_model
from defenses import create_defense
from load_datasets import load_dataset
from utils.common_utils import load_config, seed_everything, setup_device, setup_logging


def main():
    parser = argparse.ArgumentParser(description="ClipInspector backdoor detection")
    parser.add_argument("--output_dir", required=True, type=Path, help="Trained model run directory")
    parser.add_argument("--config", default="configs/defenses/clip_inspector.yaml")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    original = load_config(args.output_dir / "config.yaml")
    config = load_config(args.config)
    method = config["defense_type"]
    output_dir = args.output_dir / method / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir.mkdir(parents=True)
    logger = setup_logging(output_dir, f"detection_{method}.log")
    device = setup_device(args.gpu_id)
    seed_everything(original["seed"])

    data_config = dict(original, subsample="all", batch_size=config["batch_size"])
    _, _, test_loader, classnames, _ = load_dataset(data_config)
    # The benchmark's capped candidate set includes the known target, when available.
    max_classes = config.get("max_classes", 50)
    if len(classnames) > max_classes:
        target_idx = original.get("trigger", {}).get("target_class")
        if target_idx is None:
            classnames = random.sample(classnames, max_classes)
        else:
            _, _, _, training_classes, _ = load_dataset(original)
            target_name = training_classes[target_idx]
            others = [name for name in classnames if name != target_name]
            classnames = [target_name] + random.sample(others, max_classes - 1)
    logger.info("Detecting on %d classes from %s", len(classnames), original["dataset_name"])
    model = load_trained_model(args.output_dir, config, classnames, device)
    details = {"test_loader": test_loader, "detection_classnames": classnames}
    seed_everything(config["seed"])
    create_defense(method, model, details, config, device, output_dir).run_detection()


if __name__ == "__main__":
    main()
