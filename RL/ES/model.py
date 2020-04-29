import torch
from torch import nn
import numpy as np

class Controller(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation):
        super().__init__()
        self.num_layers = len(hidden_dims)+1
        h1d, h2d = hidden_dims
        #enforces order on parameters
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, h1d),
            nn.Linear(h1d, h2d),
            nn.Linear(h2d, output_dim),
        ])
        if activation == 'tanh':
            act_list = [nn.Tanh()] * self.num_layers
        if activation == 'none':
            act_list = [nn.Identity()] * self.num_layers
        self.activations = nn.ModuleList(act_list)
        
    def forward(self, x):
        for i in range(self.num_layers):
            x = self.activations[i](self.layers[i](x))
        return x
        #return self.model(x)

    def flat_parameters(self):
        return np.concatenate([p.flatten().detach().numpy() for p in self.parameters()])

    def set_parameters(self, params):
        """
        Transform flat parameters from ES to dictionary form and load into controller

        Args: 
            params: parameters as a single (D,) np array 
        """
        current_idx = 0
        with torch.no_grad():
            for param in self.parameters():
                param.copy_(torch.Tensor(params[current_idx:current_idx+param.nelement()].reshape(param.shape)))
                current_idx += param.nelement()

if __name__ == "__main__":
    ctrl = Controller(8, [64,32], 28, 'tanh')
    test = np.random.normal(size=ctrl.flat_parameters().size)
    ctrl.set_parameters(test)
    assert np.allclose(test, ctrl.flat_parameters())
    