#    Copyright 2025 Kenneth Wilber (kawgit)

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from torch.nn.utils import clip_grad_norm_
import os
import time
import torch
import wandb

from checkpoint import save_checkpoint
from device import device
from settings import dataset_name, model_name
from utils import format_number

class Trainer:

    def __init__(self, criterion, optimizer, scheduler):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.wandb_id = None
        self.epoch_idx = 0
        self.batch_idx = 0

    def fit(self, model, dataloader, num_epochs):

        if self.wandb_id == None:

            wandb.init(
                project="transformer",
                name=model_name
            )

            self.wandb_id = wandb.run.id

        else:

            print(f"Resuming run with id {self.wandb_id}")

            wandb.init(
                project="transformer",
                id=self.wandb_id,
                resume="must"
            )

        model.train()

        print("Compiling model...")
        compiled_model = torch.compile(model)

        time_of_last_save = time.time()
        time_between_saves = 10

        for self.epoch_idx in range(self.epoch_idx, self.epoch_idx + num_epochs):

            for self.batch_idx, (batch_xs, batch_ys) in enumerate(dataloader):
                batch_xs = batch_xs.to(device)
                batch_ys = batch_ys.to(device)

                self.optimizer.zero_grad()

                batch_outputs = compiled_model(batch_xs)

                loss = self.criterion(batch_outputs, batch_ys)
                loss.backward()
                
                if self.epoch_idx * len(dataloader) + self.batch_idx > 3000:
                    clip_grad_norm_(compiled_model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()

                wandb.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]})

                print(f'Epoch: {self.epoch_idx} Batch: {self.batch_idx} Loss: {format_number(loss.item())} LR: {format_number(self.scheduler.get_last_lr()[0])}')

                if time.time() - time_of_last_save > time_between_saves:

                    time_of_last_save = time.time()
                    
                    print("Saving checkpoint...")
                    save_checkpoint(model, self)
            