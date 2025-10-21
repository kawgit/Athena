from torch.nn import Parameter
from torch.optim import AdamW, Muon
from torch.optim import Optimizer
from typing import Dict, Any, List, Tuple, Callable, Iterable, Optional, Type, Union
import math

from athena.kuon import Kuon
from athena.utils import linear_interpolator, scale_interpolator

# ---- Types and config ----

DynamicValue = Union[Any, Callable[[float], Any]]  # e.g., 1e-3 or lambda epoch: ...
RouteTuple   = Tuple[str, Dict[str, DynamicValue]]
OptimizerSpecs = Dict[str, Tuple[Type[Optimizer], Dict[str, Any]]]

spec: OptimizerSpecs = {
    "muon":  (Muon,  {}),
    "adamw": (AdamW, {}),
    "kuon": (Kuon, {}),
}

# Return "key" for defaults, or ("key", {"lr": ..., ...}) for per-group overrides.
def router(name: str, param: Parameter) -> Optional[Union[str, RouteTuple]]:

    if "table" in name:
        return "adamw", {"lr": 3e-2, "weight_decay": 0}

    if "sas" in name:
        return "muon", { "lr": 2e-3 }
    
    if "ffn" in name:
        return "muon", { "lr": 2e-3 }

    if "vocab_proj" in name:
        return "adamw", { "lr": 2e-4 }
    
    raise ValueError(f"No optimizer routing procedure found for param with name {name} and shape {param.shape}")

class MultiOptimizer(Optimizer):
    def __init__(self, named_params: Iterable[Tuple[str, Parameter]]):
        self.dynamic_groups: List[Dict[str, Any]] = []

        for name, param in named_params:
            assert(isinstance(name, str))
            assert(isinstance(param, Parameter))
            
            if not param.requires_grad:
                continue
            
            decision = router(name, param)
            if decision is None:
                continue
            
            if isinstance(decision, str):
                opt_key = decision
                dynamic_group: Dict[str, DynamicValue] = {}
            elif isinstance(decision, tuple) and len(decision) == 2:
                opt_key, overrides = decision
                dynamic_group = spec[opt_key][1] | overrides.copy()
            else:
                raise TypeError(f"Router must return str | (str, dict) | None, got {type(decision)}")

            assert(isinstance(opt_key, str))
            assert(isinstance(dynamic_group, dict))
            assert(opt_key in spec)

            dynamic_group["name"] = name
            dynamic_group["opt_key"] = opt_key
            dynamic_group["params"] = param
            self.dynamic_groups.append(dynamic_group)
            
        self.opts: Dict[str, Optimizer] = {}

        for opt_key, (OptCls, base_kwargs) in spec.items():
            opt_dynamic_groups: List[Dict[str, Any]] = [group for group in self.dynamic_groups if group["opt_key"] == opt_key]

            if len(opt_dynamic_groups) == 0:
                continue

            opt_static_groups = [{k: (v if not callable(v) else v(0)) for k, v in group.items()} for group in opt_dynamic_groups]
            self.opts[opt_key] = OptCls(opt_static_groups)
        
        self.param_groups: List[Dict[str, Any]] = []
        for opt in self.opts.values():
            self.param_groups += list(opt.param_groups)

    def _apply_dynamic(self, epoch: float) -> None:
        if not isinstance(epoch, (int, float)):
            raise TypeError(f"epoch must be a float, got {type(epoch)}")
        epoch = float(epoch)
        if not math.isfinite(epoch):
            raise ValueError(f"epoch must be finite, got {epoch}")

        for dynamic_group in self.dynamic_groups:
            for k, v in dynamic_group.items():
                if not callable(v):
                    continue
                for param_group in self.opts[dynamic_group["opt_key"]].param_groups:
                    if param_group["name"] == dynamic_group["name"]:
                        param_group[k] = v(epoch)
                        break
                else:
                    raise KeyError(f"Param group corresponding to dynamic group {dynamic_group["name"]} not found")

    def step(self, epoch: float):
        self._apply_dynamic(epoch)
        for opt in self.opts.values():
            opt.step()

    def zero_grad(self, set_to_none: bool = False):
        for opt in self.opts.values():
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "optimizers": {k: opt.state_dict() for k, opt in self.opts.items()},
            "spec": {k: (cls.__name__, kwargs) for k, (cls, kwargs) in spec.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if "optimizers" not in state_dict:
            raise ValueError("Invalid state_dict: missing 'optimizers'.")
        for k, sub in state_dict["optimizers"].items():
            if k not in self.opts:
                raise KeyError(f"State dict contains key '{k}' not present in this MultiOptimizer.")
            self.opts[k].load_state_dict(sub)

        self.param_groups = []
        for opt in self.opts.values():
            self.param_groups += list(opt.param_groups)

    @property
    def state(self):
        return {k: opt.state for k, opt in self.opts.items()}

    def __repr__(self):
        inner = ", ".join(f"{k}={repr(opt)}" for k, opt in self.opts.items())
        return f"MultiOptimizer({inner})"
