"""Quick test to demonstrate the new training progress display."""
import time
from tqdm import tqdm

def simulate_training():
    """Simulate training to show progress bars"""
    epochs = 5
    batches_per_epoch = 100
    
    print("\n" + "="*60)
    print("Training Simulation - New Progress Display")
    print("="*60 + "\n")
    
    # Epoch progress bar
    epoch_pbar = tqdm(range(epochs), desc="Training Progress", unit="epoch", position=0)
    
    for epoch in epoch_pbar:
        total_loss = 0
        
        # Batch progress bar
        batch_pbar = tqdm(
            range(batches_per_epoch),
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
            position=1
        )
        
        for batch in batch_pbar:
            # Simulate training
            time.sleep(0.01)
            loss = 2.5 - (epoch * 0.3) - (batch * 0.001)
            total_loss += loss
            
            # Update batch progress
            batch_pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{total_loss/(batch+1):.4f}'
            })
        
        avg_loss = total_loss / batches_per_epoch
        
        # Update epoch progress
        epoch_pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'batches': batches_per_epoch
        })
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60 + "\n")

if __name__ == "__main__":
    simulate_training()
