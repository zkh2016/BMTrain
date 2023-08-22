from typing import Generator, Iterable, List, Tuple
import torch
from .block_layer import CheckpointBlock
from .parameter import DistributedParameter
from .global_var import config
from . import nccl


def init_distributed_parameter(params : Iterable[torch.nn.Parameter]):
    for param in params:
        if not isinstance(param, DistributedParameter):
            continue
        if param._init_method is None:
            continue
        with torch.no_grad():
            partition_size = param.storage().size()
            if param._tp_mode:
                world_size = config['tp_zero_size'] * (config['tp_size'] if param._tp_split_dim >= 0 else 1)
                rank =  config['tp_zero_rank'] + config['topology'].tp_id * config['tp_zero_size']
            else:
                world_size = config['zero_size']
                rank = config['zero_rank']

            global_size = partition_size *  world_size          
            tmp_storage = param.storage_type()(global_size)
            tmp_tensor = torch.tensor([], dtype=param.dtype, device="cuda")
            tmp_tensor.set_(tmp_storage, 0, param._tp_original_shape)

            param._init_method(tmp_tensor)
            if param._tp_mode:
                if param._original_shape == param._tp_original_shape:
                    begin = config['tp_zero_rank']
                    end = begin + 1
                else:
                    begin = rank                 
                    end = 1 + begin 
            else:
                begin = config['zero_rank']
                end = begin + 1

            # Pytorch 1.11 changed the API of storage.__getitem__
            torch.tensor([], dtype=param.dtype, device=param.device).set_(param.storage())[:] = \
                torch.tensor([], dtype=param.dtype, device=param.device).set_(tmp_storage)[partition_size * begin : partition_size * end]
            # param.storage().copy_(tmp_storage[partition_size * config['rank'] : partition_size * (config['rank'] + 1)])

def iterate_parameters(model : torch.nn.Module):
    for kw, val in model._parameters.items():
        if hasattr(val,"_in_checkpoint_block") and val._in_checkpoint_block:
            return []
        yield val

def init_parameters(model : torch.nn.Module):
    """
    Initialize the parameters of the model by calling the init_method of the distributed parameters.
    """

    modules = model.named_modules()
    for module_prefix, module in modules:
        if isinstance(module, CheckpointBlock):
            module.init_parameters()
        else:
            init_distributed_parameter( iterate_parameters(module) )
    
    current_stream = torch.cuda.current_stream()
    config['load_stream'].wait_stream(current_stream)

def grouped_parameters(model : torch.nn.Module) -> Generator[Tuple[str, List[torch.nn.Parameter]], None, None]:
    """
    Iterate over the parameters of the model grouped by the group name.
    This is similar to `torch.nn.Module.named_parameters()` .
    """

    ret : List[torch.nn.Parameter] = {}
    for module in model.modules():
        if isinstance(module, CheckpointBlock):
            for kw, params in module.grouped_parameters():
                if kw not in ret:
                    ret[kw] = []
                ret[kw].extend(params)
        else:
            for param in module._parameters.values():
                group = None
                if isinstance(param, DistributedParameter):
                    group = param.group
                if group not in ret:
                    ret[group] = []
                ret[group].append(param)
    for kw, val in ret.items():
        yield kw, val

