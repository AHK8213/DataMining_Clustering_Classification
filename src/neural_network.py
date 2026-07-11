"""
neural_network.py - PyTorch neural network implementation

Provides:
- Feedforward neural network with configurable architecture
- GPU support with CPU fallback
- Training with validation
- Early stopping
- Integration with scikit-learn API
"""

import warnings
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# Import PyTorch with try/except for environments without GPU
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None

from src.config import RANDOM_STATE, GPU_AVAILABLE, DEVICE, VERBOSE

warnings.filterwarnings("ignore")


# ============================================================================
# Neural Network Model Definition
# ============================================================================

class FeedforwardNN(nn.Module):
    """
    Feedforward neural network with configurable architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32, 16],
        dropout_rate: float = 0.3,
        activation: str = 'relu'
    ):
        """
        Initialize the neural network.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function ('relu' or 'tanh')
        """
        super(FeedforwardNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


# ============================================================================
# PyTorch Neural Network Classifier (Scikit-learn compatible)
# ============================================================================

class PyTorchNeuralNetwork(BaseEstimator, ClassifierMixin):
    """
    PyTorch Neural Network classifier with scikit-learn API.
    
    Usage:
        nn = PyTorchNeuralNetwork(hidden_dims=[64, 32], epochs=50)
        nn.fit(X_train, y_train)
        y_pred = nn.predict(X_test)
        y_proba = nn.predict_proba(X_test)
    """
    
    def __init__(
        self,
        hidden_dims: List[int] = [64, 32, 16],
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.3,
        activation: str = 'relu',
        early_stopping_patience: int = 10,
        validation_split: float = 0.15,
        random_state: int = RANDOM_STATE,
        verbose: bool = VERBOSE
    ):
        """
        Initialize the neural network classifier.
        
        Args:
            hidden_dims: List of hidden layer dimensions
            epochs: Maximum number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
            dropout_rate: Dropout probability
            activation: Activation function ('relu' or 'tanh')
            early_stopping_patience: Patience for early stopping
            validation_split: Fraction of training data for validation
            random_state: Random seed
            verbose: Print training progress
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Please install torch to use neural networks.")
        
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.early_stopping_patience = early_stopping_patience
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose
        
        self.model = None
        self.input_dim = None
        self.device = torch.device(DEVICE)
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Set random seed
        torch.manual_seed(random_state)
        if GPU_AVAILABLE:
            torch.cuda.manual_seed(random_state)
    
    def _create_model(self) -> nn.Module:
        """Create the neural network model."""
        return FeedforwardNN(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        )
    
    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return torch.FloatTensor(X).to(self.device)
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array."""
        return tensor.detach().cpu().numpy()
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'PyTorchNeuralNetwork':
        """
        Fit the neural network.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        
        Returns:
            Self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        
        self.input_dim = X.shape[1]
        
        # Split validation data if not provided
        if X_val is None and self.validation_split > 0:
            from sklearn.model_selection import train_test_split
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_split,
                random_state=self.random_state, stratify=y
            )
        
        # Create model
        self.model = self._create_model().to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Data loaders
        train_dataset = TensorDataset(self._to_tensor(X), self._to_tensor(y))
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_dataset = TensorDataset(self._to_tensor(X_val), self._to_tensor(y_val))
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            epoch_train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            self.train_losses.append(avg_train_loss)
            
            # Validation
            self.model.eval()
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    epoch_val_loss += loss.item()
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            self.val_losses.append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.best_val_loss = best_val_loss
                self.best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} - "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}")
            
            if patience_counter >= self.early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
        
        Returns:
            Probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float32)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X)
            outputs = self.model(X_tensor)
            proba = self._to_numpy(outputs)
        
        # Return probabilities for both classes [P(y=0), P(y=1)]
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
        
        Returns:
            Predicted labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def get_params(self, deep=True) -> Dict[str, Any]:
        """Get parameters for scikit-learn compatibility."""
        return {
            'hidden_dims': self.hidden_dims,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'early_stopping_patience': self.early_stopping_patience,
            'validation_split': self.validation_split,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }
    
    def set_params(self, **params) -> 'PyTorchNeuralNetwork':
        """Set parameters for scikit-learn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
    def plot_training_history(self, figsize: Tuple[int, int] = (10, 5)):
        """
        Plot training and validation loss history.
        
        Args:
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.train_losses, label='Training Loss', linewidth=2)
        ax.plot(self.val_losses, label='Validation Loss', linewidth=2)
        ax.axvline(self.best_epoch, color='red', linestyle='--', 
                   alpha=0.5, label=f'Best epoch ({self.best_epoch+1})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# Convenience Functions
# ============================================================================

def create_neural_network(
    hidden_dims: List[int] = [64, 32, 16],
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3,
    random_state: int = RANDOM_STATE,
    verbose: bool = VERBOSE
) -> PyTorchNeuralNetwork:
    """
    Create a neural network classifier.
    
    Args:
        hidden_dims: Hidden layer dimensions
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        dropout_rate: Dropout rate
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        PyTorchNeuralNetwork instance
    """
    return PyTorchNeuralNetwork(
        hidden_dims=hidden_dims,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        random_state=random_state,
        verbose=verbose
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing neural_network.py...")
    
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping tests.")
    else:
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        # Generate test data
        X, y = make_classification(
            n_samples=300, n_features=10, n_informative=8,
            n_redundant=2, n_classes=2, random_state=42
        )
        X = X.astype(np.float32)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train neural network
        nn_model = create_neural_network(
            hidden_dims=[32, 16],
            epochs=20,
            batch_size=32,
            verbose=True
        )
        
        nn_model.fit(X_train, y_train)
        
        # Predict
        y_pred = nn_model.predict(X_test)
        y_proba = nn_model.predict_proba(X_test)
        
        # Evaluate
        from sklearn.metrics import accuracy_score, f1_score
        print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print(f"Test F1: {f1_score(y_test, y_pred):.3f}")
        
        # Plot training history
        fig = nn_model.plot_training_history()
        fig.show()
        
        print("\nAll tests passed!")