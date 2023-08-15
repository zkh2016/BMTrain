import torch
import torch.nn.functional as F
import bmtrain as bmt
from .. import C
def c_linear(x, weight, bias, out, trans_a=False, trans_b=False) -> None:
    M = x.size(0)
    if x.dim() == 3:
        M = M * x.shape[1]
    K = weight.shape[1]
    N = weight.shape[0]
    C.linear_launcher(
            x.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr() if bias is not None else None,
            out.data_ptr(),
            M,
            K,
            N,
            trans_a,
            trans_b,
            torch.cuda.current_stream().cuda_stream
    )

def c_linear_backward(x, weight, bias, out, dout, dx, dweight, dbias, trans_a=False, trans_b=False) -> None:
    M = x.size(0)
    if x.dim() == 3:
        M = M * x.shape[1]
    K = weight.shape[1]
    N = weight.shape[0]
    C.linear_backward_launcher(
            x.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr(),
            out.data_ptr(),
            dout.data_ptr(),
            dx.data_ptr(),
            dweight.data_ptr(),
            dbias.data_ptr(),
            M,
            K,
            N,
            trans_a,
            trans_b,
            torch.cuda.current_stream().cuda_stream
    )


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


class ActivationLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, x):
        ctx.save_for_backward(x)
        ctx.module = module
        c_linear(x, module.weight, module.bias, module.out, trans_b=True)
        return module.out

    @staticmethod
    def backward(ctx, dout):
        x,  = ctx.saved_tensors
        module = ctx.module
        dx = torch.empty_like(x)
        c_linear_backward(x, module.weight, module.bias, module.out, dout, dx, module.weight.grad, module.bias.grad) 
        return None, dx

class ActivationLinear(bmt.DistributedModule):
    def __init__(self, in_features : int, out_features: int, bias: bool = True, dtype = None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = bmt.DistributedParameter(torch.empty(out_features, in_features, dtype=dtype, device="cuda"), init_method=torch.nn.init.xavier_normal_)
        if bias:
            self.bias = bmt.DistributedParameter(torch.empty(out_features, dtype=dtype, device="cuda"), init_method=torch.nn.init.zeros_)
        else:
            self.register_parameter('bias', None)
        self.out = None
    
    def forward(self, x):
        if self.out is None:
            shape = x.shape[:-1] + (self.out_features,)
            self.out = torch.empty(shape, dtype=x.dtype, device=x.device)
        return ActivationLinearFunc.apply(self, x)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

