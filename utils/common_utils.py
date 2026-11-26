import os
import random
import logging
import torch
import yaml
import numpy as np
from datetime import datetime


def seed_everything(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device(gpu_id):
    """Setup device for training, handling CUDA_VISIBLE_DEVICES properly."""
    if torch.cuda.is_available():
        # Check how many GPUs are visible to PyTorch
        num_gpus = torch.cuda.device_count()
        
        if num_gpus == 0:
            device = "cpu"
        elif gpu_id >= num_gpus:
            # If requested GPU ID is not available, use the first available GPU
            device = f"cuda:0"
            print(f"Warning: GPU {gpu_id} not available. Using cuda:0 instead. Available GPUs: {num_gpus}")
        else:
            # Use the requested GPU ID (which maps to visible devices)
            device = f"cuda:{gpu_id}"
        
        if device != "cpu":
            torch.cuda.set_device(device)
            print(f"Using device: {device} (GPU {torch.cuda.get_device_name(device)})")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")
    
    return device


def setup_logging(output_dir, log_filename="training.log"):
    """Setup logging configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(output_dir, log_filename))
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def setup_environment(config, base_output_dir, gpu_id):
    """Setup complete environment."""
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.join(base_output_dir,config['dataset_name']), f"{config['attack_type']}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed
    seed_everything(config['seed'])
    
    
    # Setup device
    device = setup_device(gpu_id)
    
    # Setup logging
    logger = setup_logging(output_dir)
    
    logger.info(f"Environment setup complete")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {config['seed']}")
    
    return device, logger, output_dir


def save_model_and_results(model, trigger_info, clean_acc, asr, config, output_dir):
    """Save model and results."""
    import yaml
    import json
    
    # Save model checkpoint
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {
        'prompt_learner_state_dict': model_to_save.prompt_learner.state_dict(),
        'config': config,
        'clean_accuracy': clean_acc,
        'attack_success_rate': asr
    }
    
    torch.save(checkpoint, os.path.join(output_dir, 'model_checkpoint.pth'))
    
    torch.save(trigger_info, os.path.join(output_dir, 'trigger_info.pth'))
    
    # Save results summary
    results = {
        'attack_type': config['attack_type'],
        'dataset_name': config['dataset_name'],
        'clean_accuracy': float(clean_acc),
        'attack_success_rate': float(asr),
        'config': config
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)