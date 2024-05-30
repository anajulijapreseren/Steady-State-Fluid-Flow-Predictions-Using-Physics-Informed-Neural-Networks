import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class LBFGS_Optimizer:
    def __init__(self, model, x_train, y_train, max_iter=20000, history_size=50, lr=1):
        self.model = model
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.max_iter = max_iter
        self.history_size = history_size
        self.lr = lr
        self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=self.lr, 
                                           max_iter=20000, history_size=self.history_size)

    def step(self):
        # Define closure function for LBFGS optimization
        def closure():
            if torch.is_grad_enabled():
                self.optimizer.zero_grad()
            outputs = self.model(self.x_train)
            loss = self.model.loss_function(outputs, self.y_train)
            if loss.requires_grad:
                loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure().item()  # Return the loss value

    def fit(self):
        for _ in range(self.max_iter):
            loss = self.step()
            print(f'Current Loss: {loss}')

