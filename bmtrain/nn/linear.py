import torch
import torch.nn.functional as F
import bmtrain as bmt

class CustomLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward(x, weight, bias)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        if x.requires_grad:
            grad_x = grad_output.matmul(weight)
        if weight.requires_grad:
            dim = grad_output.dim()
            grad_weight = grad_output.reshape(-1,
                grad_output.shape[-1]).t().matmul(x.reshape(-1, x.shape[-1]))
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_x, grad_weight, grad_bias

class Linear(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=dtype, device="cuda"), init_method=torch.nn.init.xavier_normal_)
        if bias:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=dtype, device="cuda"), init_method=torch.nn.init.zeros_)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input):
        return CustomLinear.apply(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

from .. import C
def c_linear(x, weight, bias, out) -> None:
    C.linear_launcher(
            x.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr(),
            out.data_ptr(),
            x.size(0),
            weight.size(0),
            weight.size(1),
            torch.cuda.current_stream().cuda_stream
    )

#class BMTLinearFunction(torch.autograd.Function):
    
    
class BMTLinear:
    def __init__(self):
        self.out_dict = {}
    def run(self, x, weight, bias):
        if x.dim() == 3:
            x = x.view(x.shape[0] * x.shape[1], x.shape[2])
        out_shape = (x.shape[0], weight.shape[1])
        if out_shape in self.out_dict:
            out_tensor = self.out_dict[out_shape]
        else:
            out_tensor = torch.empty(out_shape, dtype=x.dtype, device=x.device)
        c_linear(x, weight, bias, out_tensor)
        return out_tensor
