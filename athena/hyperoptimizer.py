import numpy as np
import math
from torch.optim import AdamW

from athena.graft import graft
from athena.model import Athena
from athena.pretrain import Pretrainer
from athena.utils import generate_model_name, grow_nice_number

class AthenaHyperoptimizer():
    
    def __init__(self, init_model_config, mutation_num=10, mutation_time=30, mutatable_keys=["embedding_size", "hidden_size", "head_size", "num_heads", "num_layers"]):
        self.main_configs = [init_model_config]
        self.mutation_num = mutation_num
        self.mutation_time = mutation_time
        self.mutatable_keys = mutatable_keys
        
        num_possible_mutations = 2 ** len(self.mutatable_keys)
        self.mutation_probs = np.array([1 / num_possible_mutations] * num_possible_mutations)
        
    def train(self, steps=100):
        
        for i in range(steps):
            self.step()
            
    def step(self):
        
        mutations = np.random.choice(range(len(self.mutation_probs)), self.mutation_num, p=self.mutation_probs, replace=False)
        results = []

        for mutation in mutations:
            
            mutated_config = self.main_configs[-1].copy()
            
            for i in range(len(self.mutatable_keys)):
                if mutation & (1 << i):
                    mutated_config[self.mutatable_keys[i]] = grow_nice_number(mutated_config[self.mutatable_keys[i]])
            
            mutated_config["name"] = generate_model_name(mutated_config)
            mutated_valid_loss = self.step_run(mutated_config)
            
            results.append((mutation, mutated_config, mutated_valid_loss))

        self.mutation_probs = []
        
        results.sort(key=lambda x: x[2])
        
        self.main_configs.append(results[0][1])

    def step_run(self, mutated_config):
        
        main_athena = Athena(self.main_configs[-1])
        mutated_athena = Athena(mutated_config)
        graft(main_athena, mutated_athena)
        
        optimizer = AdamW(mutated_athena.parameters(), lr=3e-5)
        
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=50)
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=950)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, main_sched], milestones=[50])
        
        pretrainer = Pretrainer(mutated_athena, optimizer, scheduler)
        pretrainer.train(epoch_limit=float("inf"), time_limit=self.mutation_time)

        # return pretrainer.run.summary["valid_loss"]

