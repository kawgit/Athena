from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ConstantLR, SequentialLR

from athena.checkpoint import load_checkpoint
from athena.graft import graft
from athena.model import Athena
from athena.pretrain import Pretrainer
from athena.utils import generate_model_name, grow_nice_number

class AthenaHyperoptimizer():
    
    def __init__(self, init_model_config, step_time=30, mutatable_keys=["embedding_size", "hidden_size", "head_size", "num_heads", "num_layers"]):
        self.config_history = [init_model_config]
        self.step_time = step_time
        self.mutatable_keys = mutatable_keys
        self.mutation_grad = { key: 0 for key in self.mutatable_keys }
        
    def train(self, steps=100):
        
        for i in range(steps):
            self.step()
            
    def step(self):
        
        for key in self.mutatable_keys:
            
            mutated_config = self.config_history[-1]
            mutated_config[key] = grow_nice_number(mutated_config[key])
            mutated_config["name"] = generate_model_name(mutated_config)
            
            self.step_internal(mutated_config)

    def step_internal(self, mutated_config):
        
        
        main_config = self.config_history[-1]
        main_athena = load_checkpoint(main_config.name)[0]
        mutated_athena = Athena(mutated_config)
        graft(main_athena, mutated_athena)
        
        optimizer = AdamW(mutated_athena.parameters(), lr=3e-5)
        
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=50)
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=950)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[50])
        
        pretrainer = Pretrainer(mutated_athena, optimizer, scheduler)
        pretrainer.train(epoch_limit=float("inf"), time_limit=self.mutation_time)

        # return pretrainer.run.summary["valid_loss"]

