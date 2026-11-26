import logging
from tqdm import tqdm

from .base_attack import BaseAttack


class CleanTraining(BaseAttack):
    """Clean training without any backdoor."""
    
    def __init__(self, model, config, device, output_dir=None):
        super().__init__(model, config, device, output_dir)
        self.logger.info("Initializing clean training")
    
    def create_triggered_batch(self, images, labels):
        """No trigger for clean training."""
        return images, labels
    
    def should_use_trigger(self, batch_idx, epoch):
        """Never use trigger in clean training."""
        return False
    
    def get_trigger_info(self):
        """Return empty trigger info for clean training."""
        return {
            'attack_type': 'clean',
            'target_class': None,
            'trigger_data': None,
            'config': self.config,
            'metadata': {'description': 'Clean training without backdoor'}
        }
    
    def train(self, train_loader, val_loader=None):
        """Train the model cleanly."""
        self.logger.info("Starting clean training...")
        
        # Track metrics for plotting
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.config['epochs']):
            train_loss, val_loss, val_acc,_,_ = self.train_epoch(epoch, train_loader, val_loader)
            
            # Store metrics
            train_losses.append(train_loss)
            if val_loss is not None:
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)
            
            # Log progress
            log_msg = (f"Epoch {epoch+1}/{self.config['epochs']} - "
                      f"Train Loss: {train_loss:.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if val_loss is not None:
                log_msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            
            self.logger.info(log_msg)
        
        self.logger.info("Clean training completed")
        
        # Save training plots if we have validation data
        if val_losses:
            self.save_training_plots(train_losses, val_losses, val_accuracies)
        
        # Return model and trigger info
        return self.model, self.get_trigger_info()
    
    def save_training_plots(self, train_losses, val_losses, val_accuracies):
        """Save training progress plots."""
        try:
            import matplotlib.pyplot as plt
            import os
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss plot
            epochs = range(1, len(train_losses) + 1)
            ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
            if val_losses:
                ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Progress - Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy plot
            if val_accuracies:
                ax2.plot(epochs, val_accuracies, 'g-', label='Val Accuracy')
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