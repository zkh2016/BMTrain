import torch
import torch.nn.functional as F
import bmtrain as bmt
from bmtrain.nn.linear import (
        c_linear,
        c_linear_backward
    )
from bmtrain.nn import Linear, ActivationLinear 

device=torch.device('cuda')

bs = 2
seq_l = 16
in_f = 256
out_f = 512

x = torch.randn(bs, seq_l, in_f, device=device, dtype=torch.bfloat16).requires_grad_()
weight = torch.randn(in_f, out_f, device=device, dtype=torch.bfloat16)
bias = torch.randn(out_f, device=device, dtype=torch.bfloat16)
out = torch.randn(bs, seq_l, out_f, device=device, dtype=torch.bfloat16)
dx = torch.randn(bs, seq_l, in_f, device=device, dtype=torch.bfloat16)
dweight = torch.randn(in_f, out_f, device=device, dtype=torch.bfloat16)
dbias = torch.randn(out_f, device=device, dtype=torch.bfloat16)

linear = torch.nn.Linear(in_f, out_f, device=device, dtype=torch.bfloat16, bias=True)
y = linear(x)
dout = torch.rand_like(y, device='cuda')
y.backward(dout)
print(y.sum())
print(x.grad.sum())
print(linear.weight.grad.sum())
print(linear.bias.grad.sum())
print()

c_linear(x, linear.weight, linear.bias, out, trans_b=True)
print(out.sum())
c_linear_backward(x, linear.weight, linear.bias, out, dout, dx, dweight, dbias)
print(dx.sum())
print(dweight.sum())
print(dbias.sum())
print()

#bmt.init_distributed(zero_level=3)
#linear1 = bmt.CheckpointBlock(ActivationLinear(4,4, dtype=torch.bfloat16, bias=True), use_checkpoint=False)
#bmt.init_parameters(linear1)
#y = linear1(x)
#print(x.grad)
#print(y)
#dout = torch.randn(y.shape, device='cuda')
#print(dout)
#y.backward(dout)
#print(linear1.weight.grad)
#print(x.grad.data_ptr())
#
