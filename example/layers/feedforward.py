import torch
import bmtrain as bmt
from bmtrain.nn import (
        Linear,
        ActivationLinear
)

class Feedforward(bmt.DistributedModule):
    def __init__(self, dim_model : int, dim_ff : int, bias : bool = True, dtype = None) -> None:
        super().__init__()

        if True:
            self.w_in = ActivationLinear(dim_model, dim_ff, bias = bias, dtype=dtype)
            self.w_out = ActivationLinear(dim_ff, dim_model, bias = bias, dtype=dtype)
            self.gate = ActivationLinear(dim_model, dim_ff, bias = bias, dtype=dtype)
        else:
            self.w_in = Linear(dim_model, dim_ff, bias = bias, dtype=dtype)
            self.w_out = Linear(dim_ff, dim_model, bias = bias, dtype=dtype)
            self.gate = Linear(dim_model, dim_ff, bias = bias, dtype=dtype)

        self.relu = torch.nn.ReLU()
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        gate_out = self.relu(self.gate(input))

        return self.w_out(self.w_in(input) * gate_out)
