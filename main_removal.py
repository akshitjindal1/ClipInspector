# main_removal.py
import argparse
import json
import csv
import math
from datetime import datetime
from pathlib import Path
import sys

import torch
import yaml

from classes import load_trained_model
from load_datasets import load_dataset
from removal import create_remover
from removal.trigger_appliers import AdditiveTriggerApplier
from utils.common_utils import load_config, seed_everything, setup_device, setup_logging
from utils.eval_utils import evaluate_attack_success_rate, evaluate_clean_accuracy


PROJECT_ROOT = Path(__file__).resolve().parent


def load_detection_target(detection_dir):
    summary = detection_dir / "backdoor_detection_summary.txt"
    lines = summary.read_text().splitlines()
    losses = [line.strip() for line in lines if line.startswith("Loss function:")]
    if losses != ["Loss function: margin"]:
        raise ValueError(
            f"{summary} does not identify a single margin reconstruction. "
            "Rerun margin-only detection before removal."
        )

    header = next((i for i, line in enumerate(lines) if line.startswith("Class\t")), None)
    if header is None:
        raise ValueError(f"No per-class detection table in {summary}")
    table = []
    for row in csv.DictReader(lines[header:], delimiter="\t"):
        if not row["Class"].isdigit():
            break
        score = float(row["Anomaly Score"])
        if math.isfinite(score):
            table.append({"idx": int(row["Class"]), "name": row["ClassName"],
                          "asr": float(row["ASR (%)"]), "score": score})
    if not table:
        raise ValueError(f"No finite anomaly scores in {summary}")
    return max(table, key=lambda entry: entry["score"])


def build_removal_output_dir(detection_dir: Path, method: str, timestamp: str) -> Path:
    detection_dir = detection_dir.resolve()
    parts = detection_dir.parts
    relative_path = None
    if "outputs" in parts:
        idx = parts.index("outputs")
        relative_path = Path(*parts[idx:])
    else:
        try:
            relative_path = detection_dir.relative_to(PROJECT_ROOT)
        except ValueError:
            relative_path = Path(detection_dir.name)
    removal_root = PROJECT_ROOT / "removal_logs_cleanonly" / relative_path
    return removal_root / f"removal_{method}" / timestamp


def build_trigger_applier(detection_dir, detection_idx, removal_config):
    image_size = removal_config["image_size"]
    recon_root = detection_dir / "reconstructed_triggers"
    trigger_path = recon_root / f"recon_trigger_class_{detection_idx}.pt"
    if not trigger_path.exists():
        raise FileNotFoundError(f"Reconstructed trigger missing: {trigger_path}")
    trigger = torch.load(trigger_path, map_location="cpu")
    if isinstance(trigger, dict) and "trigger" in trigger:
        trigger = trigger["trigger"]
    return AdditiveTriggerApplier(trigger, image_size), trigger_path.name


def save_removal_results(model, clean_acc, asr, config, output_dir, metadata=None):
    metadata = metadata or {}
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint = {
        "prompt_learner_state_dict": model_to_save.prompt_learner.state_dict(),
        "config": config,
        "clean_accuracy": clean_acc,
        "attack_success_rate": asr,
    }
    torch.save(checkpoint, output_dir / "model_checkpoint.pth")
    results = {
        "clean_accuracy": float(clean_acc),
        "attack_success_rate": float(asr),
        "remover_type": config["remover_type"],
        **metadata,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def main():
    parser = argparse.ArgumentParser(description="Backdoor removal with fixed trigger")
    parser.add_argument(
        "--recon_dir",
        type=str,
        required=True,
        help="Path to the ClipInspector run containing reconstructed triggers",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/removers/fixed_trigger.yaml",
        help="Removal configuration file",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU to use",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default=None,
        help="Optional path to write per-run summary JSON",
    )
    args = parser.parse_args()

    detection_dir = Path(args.recon_dir).absolute()
    if not detection_dir.exists():
        raise FileNotFoundError(f"Recon directory not found: {detection_dir}")
    attack_run_dir = detection_dir.parent.parent
    original_config_path = attack_run_dir / "config.yaml"
    if not original_config_path.exists():
        raise FileNotFoundError(f"Missing original config at {original_config_path}")

    original_config = load_config(str(original_config_path))
    removal_config = {**original_config, **load_config(args.config)}
    for key in ("dataset_name", "data_root", "backbone", "n_ctx", "ctx_init", "use_fp16", "use_fp32", "shots"):
        if key in original_config:
            removal_config[key] = original_config[key]
    target_entry = load_detection_target(detection_dir)
    method = removal_config["remover_type"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = build_removal_output_dir(detection_dir, method, timestamp)
    output_dir.mkdir(parents=True)
    logger = setup_logging(output_dir, f"removal_{method}.log")
    logger.info("BACKDOOR REMOVAL")
    logger.info(f"Detection directory: {detection_dir}")
    logger.info(f"Attack run directory: {attack_run_dir}")
    logger.info(
        f"Selected detection class idx {target_entry['idx']} ({target_entry['name']})"
    )
    logger.info(f"Removal logs directory: {output_dir}")

    device = setup_device(args.gpu_id)
    seed_everything(removal_config.get("seed", original_config.get("seed", 0)))

    train_loader, val_loader, test_loader, classnames, class_to_idx = load_dataset(
        removal_config
    )
    if target_entry["name"] not in class_to_idx:
        logger.warning(
            "Target class '%s' not present in removal dataset (subsample=%s). "
            "Reloading dataset with 'all' subsample.",
            target_entry["name"],
            removal_config.get("subsample"),
        )
        removal_config["subsample"] = "all"
        train_loader, val_loader, test_loader, classnames, class_to_idx = load_dataset(
            removal_config
        )
        if target_entry["name"] not in class_to_idx:
            raise ValueError(
                f"Target class '{target_entry['name']}' not found in removal dataset "
                f"even after reloading with 'all' subsample."
            )
    removal_target_idx = class_to_idx[target_entry["name"]]
    removal_config["target_class_name"] = target_entry["name"]

    trigger_applier, trigger_artifact = build_trigger_applier(
        detection_dir, target_entry["idx"], removal_config
    )
    logger.info("Using margin reconstruction %s", trigger_artifact)
    model = load_trained_model(attack_run_dir, removal_config, classnames, device)

    trigger_info_path = attack_run_dir / "trigger_info.pth"
    saved_trigger_info = None
    can_evaluate_asr = False
    if trigger_info_path.exists():
        saved_trigger_info = torch.load(
            trigger_info_path, map_location="cpu", weights_only=False
        )
        can_evaluate_asr = callable(saved_trigger_info.get("metadata", {}).get("apply_function"))
    if not can_evaluate_asr:
        logger.info("Original trigger unavailable; skipping pre- and post-removal ASR.")

    remover = create_remover(
        removal_config["remover_type"],
        model,
        removal_config,
        device,
        output_dir,
        trigger_applier,
    )

    logger.info("Evaluating baseline (pre-removal) metrics...")
    clean_acc_before = evaluate_clean_accuracy(model, test_loader, device)
    logger.info(f"Clean accuracy before removal: {clean_acc_before:.4f}")
    asr_before = None
    if can_evaluate_asr:
        asr_before = evaluate_attack_success_rate(
            model, saved_trigger_info, test_loader, device
        )
        logger.info(f"ASR before removal: {asr_before:.4f}")

    logger.info("Starting removal fine-tuning...")
    remover.train(train_loader, val_loader)

    logger.info("Evaluating clean accuracy after removal...")
    clean_acc_after = evaluate_clean_accuracy(model, test_loader, device)
    logger.info(
        f"Clean accuracy after removal: {clean_acc_after:.4f} "
        f"(delta {clean_acc_after - clean_acc_before:+.4f})"
    )

    asr_after = None
    if can_evaluate_asr:
        logger.info("Evaluating ASR after removal...")
        asr_after = evaluate_attack_success_rate(
            model, saved_trigger_info, test_loader, device
        )
        delta_asr = (
            asr_after - asr_before if asr_before is not None else float("nan")
        )
        logger.info(
            f"ASR after removal: {asr_after:.4f} "
            f"(delta {delta_asr:+.4f})"
        )

    metadata = {
        "target_class_name": target_entry["name"],
        "target_detection_idx": int(target_entry["idx"]),
        "removal_dataset_idx": int(removal_target_idx),
        "defense_type": "clip_inspector",
        "detection_source": "summary:backdoor_detection_summary.txt",
        "trigger_artifact": trigger_artifact,
        "clean_accuracy_before": float(clean_acc_before),
        "clean_accuracy_after": float(clean_acc_after),
    }
    metadata["detected_class_asr_percent"] = target_entry["asr"]
    metadata["detected_class_anomaly_score"] = target_entry["score"]
    if asr_before is not None:
        metadata["asr_before"] = float(asr_before)
    if asr_after is not None:
        metadata["asr_after"] = float(asr_after)

    save_removal_results(
        model,
        clean_acc_after,
        asr_after if asr_after is not None else 0.0,
        removal_config,
        output_dir,
        metadata=metadata,
    )

    run_summary = {
        "dataset": removal_config.get("dataset_name"),
        "defense_type": "clip_inspector",
        "detection_dir": str(detection_dir),
        "attack_run_dir": str(attack_run_dir),
        "removal_output_dir": str(output_dir),
        "target_detection_idx": int(target_entry["idx"]),
        "target_class_name": target_entry["name"],
        "clean_accuracy_before": clean_acc_before,
        "clean_accuracy_after": clean_acc_after,
        "clean_accuracy_delta": clean_acc_after - clean_acc_before,
        "asr_before": asr_before,
        "asr_after": asr_after,
        "asr_delta": (asr_after - asr_before) if (asr_before is not None and asr_after is not None) else None,
        "detection_source": "summary:backdoor_detection_summary.txt",
        "trigger_artifact": trigger_artifact,
        "method": method,
        "timestamp": timestamp,
        "recon_dir": str(detection_dir),
        "command": " ".join(sys.argv),
    }
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(run_summary, f, indent=2)

    logger.info("Removal completed successfully.")
    logger.info(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
