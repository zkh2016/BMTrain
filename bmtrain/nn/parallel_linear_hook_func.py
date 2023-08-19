import torch
import torch.nn.functional as F
from bmtrain.global_var import config 
from ..distributed import all_gather, all_reduce
from .. import nccl

class LinearHookFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, gather_output=False):
        ctx.save_for_backward(input, weight, bias)
        ctx.gather_output = gather_output
        #gather input
        all_input = all_gather(input, config['tp_comm'])
        all_input = all_input.flatten(0, 1)
        out = F.linear(all_input, weight, bias)
        if gather_output:
            all_output_list = all_gather(out, config['tp_comm'])
            all_output_list = all_output_list.chunk(config['tp_size'], dim=0)        
            out = torch.cat(all_output_list, dim=all_output_list[0].dim()-1).flatten(0,1)
        return out 
            
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        gather_output = ctx.gather_output
        if gather_output:
            tp_size = config['tp_size']
            tp_id = config['topology'].tp_id
            grad_output_list = grad_output.chunk(tp_size, dim=1)
            grad_output = grad_output_list[tp_id]

        grad_input = grad_weight = grad_bias = None

        if input.requires_grad or weight.requires_grad:
            all_input = all_gather(input, config['tp_comm'])
            all_input = all_input.flatten(0, 1)

        if input.requires_grad:
            #gather can async with grad_out.matmul(weight)
            #TODO: gather on load_stream
            grad_all_input = grad_output.matmul(weight)
            grad_input = torch.empty_like(input)
            
            nccl.reduceScatter(grad_all_input.storage(), grad_input.storage(), "sum", config['tp_comm'])

        if weight.requires_grad:
            dim = grad_output.dim()
            grad_weight = grad_output.reshape(-1,
                grad_output.shape[-1]).t().matmul(all_input.reshape(-1, all_input.shape[-1]))
         
        if bias is not None and bias.requires_grad:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        return grad_input, grad_weight, grad_bias, None
