import torch.nn.functional as functional
import wandb
from torch.amp import GradScaler, autocast
from torch.utils.data import Subset

from athena.checkpoint import save_checkpoint
from athena.dataloader import load_dataloader_pretrain
from athena.device import device
from athena.model import AthenaCompiled
from athena.utils import Throttle, Timer

class Pretrainer():
    
    def __init__(self, athena, optimizer, scheduler, batch_size=4, save_every=120, valid_every=float("inf"), epoch_limit=1, time_limit=float("inf")):
        self.athena = athena
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.athena_compiled = AthenaCompiled(athena)
        self.scaler = GradScaler(device.type)
        self.batch_size = batch_size
        self.save_every = save_every
        self.valid_every = valid_every
        
    def train(self, epoch_limit, time_limit=float("inf")):
        
        self.athena.update_config(wandb_id=self.athena.wandb_id or wandb.util.generate_id())
        self.run = wandb.init(
            project="athena-pretrain",
            id=self.athena.wandb_id,
            name=self.athena.name,
            resume="allow",
            config={
                "batch_size": self.batch_size,
                **self.athena.config,
            }
        )
        
        try:
            self.train_internal(epoch_limit, time_limit)
                                        
        except KeyboardInterrupt:
            pass
        
        wandb.finish()

        with Timer("Saving checkpoint"):
            save_checkpoint(self.athena, self.optimizer, self.scheduler)
        
    def train_internal(self, epoch_limit, time_limit):
        
        save_throttle = Throttle(f"Saving checkpoint", self.save_every)
        valid_throttle = Throttle(f"Validating", self.valid_every)
            
        while True:
                    
            train_dataloader, valid_dataloader = load_dataloader_pretrain(self.athena.context_size + 1, self.batch_size, resume_epoch=self.run.summary.get("epoch", 0))

            print("Starting epoch...")

            for batch in train_dataloader:
                
                step_info = self.step(batch)
                step_info["epoch"] = step_info["step"] / len(train_dataloader)
                
                print(f"Step {step_info['step']} out of {len(train_dataloader)} "
                        f"Time {step_info['training_time']:.4f} "
                        f"Loss {step_info['loss']:.4f} "
                        f"Step Time {step_info['step_time']:.4f} "
                        f"LR {step_info['lr']:.2e}")
                
                with valid_throttle as should_run:
                    if should_run:
                        step_info["valid_loss"] = self.validate(valid_dataloader)
                        
                
                wandb.log(step_info)

                with save_throttle as should_run:
                    if should_run:
                        save_checkpoint(self.athena, self.optimizer, self.scheduler)
                        
                if self.run.summary.get("epoch", 0) >= epoch_limit or self.run.summary.get("training_time", 0) >= time_limit:
                    return
                
    def step(self, batch):
        
        step_info = {}
        self.athena_compiled.train()

        with Timer() as step_timer:
            
            step_info["step"] = self.run.summary.get("step", 0) + 1
            
            batch = batch.to(device, non_blocking=True)
            batch_x = batch[:, :-1]
            batch_y = batch[:, 1:]
            
            self.optimizer.zero_grad()

            with autocast(device.type):
                output = self.athena_compiled(batch_x)["logits"]
                loss = self.criterion(output, batch_y)

            step_info["loss"] = loss.detach().item()
            step_info["lr"] = self.optimizer.param_groups[0]['lr']
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        step_info["step_time"] = float(step_timer) if float(step_timer) < 10 * self.run.summary.get("step_time", float("inf")) else self.run.summary.get("step_time", float("inf"))
        step_info["training_time"] = self.run.summary.get("training_time", 0) + step_info["step_time"]

        if self.scheduler is not None:
            self.scheduler.step()

        return step_info

    def validate(self, valid_dataloader):

        self.athena_compiled.eval()
        valid_loss = 0
        for batch in valid_dataloader:
            batch = batch.to(device, non_blocking=True)
            batch_x = batch[:, :-1]
            batch_y = batch[:, 1:]
            
            with autocast(device.type):
                output = self.athena_compiled(batch_x)["logits"]
                valid_loss += self.criterion(output, batch_y).detach().item()
                
        valid_loss /= len(valid_dataloader)
        self.athena_compiled.train()
        
        return valid_loss

    def criterion(self, model_output, expected_output):
        
        logits = model_output.reshape(-1, model_output.shape[-1])
        labels = expected_output.reshape(-1)
        
        return functional.cross_entropy(logits, labels)