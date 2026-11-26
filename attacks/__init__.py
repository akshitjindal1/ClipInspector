from .clean_training import CleanTraining
from .badclip import BadClipAttack

def create_attack(attack_type, model, config, device, output_dir=None):
    """Factory function to create attack instances."""
    
    attack_classes = {
        'clean': CleanTraining,
        'badclip': BadClipAttack,
    }
    
    if attack_type not in attack_classes:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    return attack_classes[attack_type](model, config, device, output_dir)

