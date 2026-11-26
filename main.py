import os
import argparse
import yaml
import logging
from datetime import datetime

from load_datasets import load_dataset
from classes import initialize_clip_model
from utils.common_utils import setup_environment, save_model_and_results, load_config
from utils.eval_utils import evaluate_clean_accuracy, evaluate_attack_success_rate
from attacks import create_attack


def main():
    parser = argparse.ArgumentParser(description='Backdoor Attack Framework')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Base output directory for results')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use for training (default: 0)')
    args = parser.parse_args()
    
    # 1. Load configuration
    config = load_config(args.config)
    
    # 2. Setup environment (logging, device, seeding) and get the actual output directory
    device, logger, output_dir = setup_environment(config, args.output_dir, args.gpu_id)
    
    logger.info(f"Starting {config['attack_type']} attack")
    logger.info(f"Dataset: {config['dataset_name']}")
    logger.info(f"Device: {device}")
    
    # 3. Load dataset
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader, classnames, class_to_idx = load_dataset(config)
    
    logger.info(f"Dataset loaded: {len(classnames)} classes")
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Val samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # 4. Initialize CLIP model
    logger.info("Initializing CLIP model...")
    model = initialize_clip_model(config, classnames, device)
    
    # 5. Create attack
    logger.info(f"Creating {config['attack_type']} attack...")
    attack = create_attack(config['attack_type'], model, config, device, output_dir)
    
    # 6. Train with backdoor/clean
    logger.info("Starting training...")
    backdoored_model, trigger_info = attack.train(train_loader, val_loader)
    # import pdb;pdb.set_trace()
    
    # 7. Evaluate clean accuracy
    logger.info("Evaluating clean accuracy...")
    clean_acc = evaluate_clean_accuracy(backdoored_model, test_loader, device)
    logger.info(f"Clean Accuracy: {clean_acc:.4f}")
    
    # 8. Evaluate attack success rate (if not clean training)
    if config['attack_type'] != 'clean':
        logger.info("Evaluating attack success rate...")
        asr = evaluate_attack_success_rate(backdoored_model, trigger_info, test_loader, device)
        logger.info(f"Attack Success Rate: {asr:.4f}")
    else:
        asr = 0.0
        logger.info("Clean training - no ASR evaluation")
    
    # 9. Save results (use the same output_dir where logs are saved)
    logger.info("Saving results...")
    save_model_and_results(backdoored_model, trigger_info, clean_acc, asr, config, output_dir)
    
    logger.info("Completed successfully!")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
