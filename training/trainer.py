import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import copy

class EarlyStopping:
    """
    Early stopping helper to terminate training when validation loss stops improving.
    """
    def __init__(self, patience: int = 10, verbose: bool = True, delta: float = 0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_loss is None:
            self.best_loss = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


class Trainer:
    """
    Standard PyTorch model trainer for traffic flow prediction.
    """
    def __init__(self, model: nn.Module, device: str = 'auto', 
                 learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 grad_clip: float = 5.0, patience: int = 10, checkpoint_dir: str = 'checkpoints'):
        
        # 1. Device Configuration
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Training on device: {self.device}")
        
        self.model = model.to(self.device)
        self.grad_clip = grad_clip
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 2. Loss & Optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 3. Learning Rate Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 4. Early Stopping
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        
        for X, Y in dataloader:
            X, Y = X.to(self.device), Y.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(X)  # Shape: (batch_size, num_horizons, num_sensors)
            
            loss = self.criterion(out, Y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                
            self.optimizer.step()
            total_loss += loss.item() * X.size(0)
            
        return total_loss / len(dataloader.dataset)

    def val_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X, Y in dataloader:
                X, Y = X.to(self.device), Y.to(self.device)
                out = self.model(X)
                loss = self.criterion(out, Y)
                total_loss += loss.item() * X.size(0)
                
        return total_loss / len(dataloader.dataset)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, 
            X_val: np.ndarray, Y_val: np.ndarray,
            epochs: int = 50, batch_size: int = 64, model_name: str = 'best_model.pt'):
        """
        Fits the model on training data.
        """
        # Convert to tensors
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(Y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        train_losses = []
        val_losses = []
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Step scheduler based on validation loss
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Check early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered. Terminating training.")
                break
                
        # Load best weights
        if self.early_stopping.best_model_state is not None:
            self.model.load_state_dict(self.early_stopping.best_model_state)
            
        # Save best model to disk
        save_path = os.path.join(self.checkpoint_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"Best model saved to {save_path}")
        
        return train_losses, val_losses

    def predict(self, X: np.ndarray, batch_size: int = 64):
        """
        Generates predictions for a given input dataset.
        """
        self.model.eval()
        x_tensor = torch.tensor(X)
        dataset = TensorDataset(x_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                out = self.model(batch_x)
                preds.append(out.cpu().numpy())
                
        return np.concatenate(preds, axis=0)
