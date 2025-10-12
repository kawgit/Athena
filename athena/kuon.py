import torch
import torch.nn as nn
from typing import Any
from torch import Tensor
from torch.nn import Parameter, Parameter
from torch.optim import Optimizer, AdamW

from athena.utils import concat_or_create, lerp_or_create

def attach_kuon_hooks(model: nn.Module):

    def _forward_hook(mod: nn.Linear, args, output: torch.Tensor):
        
        param: nn.Parameter = mod.weight
        if not isinstance(param, nn.Parameter):
            print("Expected mod.weight to nn.Parameter but found", type(param))
            return
        
        if not getattr(param, "_marked_for_kuon", False):
            return
        
        def _backward_hook(grad: torch.Tensor):
            with torch.no_grad():
                x: torch.Tensor = args[0]
                x = x.detach()
                x = x.reshape(-1, x.size(-1)) # (B, I)
                g = grad.reshape(-1, grad.size(-1)) # (B, O)
                
                assert(x.size(0) == g.size(0))
                assert(len(x.shape) == 2)
                
                concat_or_create(param, "_x", x)
                concat_or_create(param, "_g", g)

                assert(param._x.size(0) == param._g.size(0))
            
        output.register_hook(_backward_hook)
            
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_forward_hook))
    return hooks

class Kuon(Optimizer):
    
    def __init__(self, params, lr=1e-1, update_lr=.001):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if update_lr < 0.0:
            raise ValueError(f"Invalid update_lr: {update_lr}")
        defaults = dict(lr=lr, update_lr=update_lr)
        super().__init__(params, defaults)
        
        for group in self.param_groups:
            for param in group['params']:
                param._marked_for_kuon = True
                param._lr = group["lr"]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                self.step_param(group, param)

        return loss

    # @torch.no_grad()
    # def step_param(self, group: dict[str, Any], param: Parameter):
        
    #     O, I = param.shape
        
    #     assert(hasattr(param, "_q"))
        
    #     lerp_or_create(param, "_v", param.grad, momentum=.99)
        
    #     # update = _zeropower_via_newtonschulz(param._v, (DEFAULT_A, DEFAULT_B, DEFAULT_C), DEFAULT_NS_STEPS, DEFAULT_NS_STEPS)
        
    #     # adamw = AdamW([param], lr=group["update_lr"])
        
    #     # adamw = 
        
    #     # update = solve_min_linear_under_quad(param.grad, param._q, group["lr"])
    #     # param += update