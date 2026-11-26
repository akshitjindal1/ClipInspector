import torch
from tqdm import tqdm


def evaluate_clean_accuracy(model, test_loader, device):
    """Evaluate clean accuracy of the model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating clean accuracy"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy
    
def evaluate_attack_success_rate(model, trigger_info, test_loader, device):
    """Measure the fraction of triggered test images predicted as the target class."""
    target_class = trigger_info['target_class']
    apply_trigger_func = trigger_info['metadata']['apply_function']
    
    successful_attacks = 0
    total_considered = 0

    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Evaluating ASR"):
            triggered_images = apply_trigger_func(images.to(device))
            logits = model(triggered_images)
            pred = logits.argmax(dim=1)

            successful_attacks += (pred == target_class).sum().item()
            total_considered += len(images)
    asr = successful_attacks / total_considered if total_considered > 0 else 0.0
    return asr
