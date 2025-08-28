import torch.nn.functional as functional
import wandb
from torch.amp import autocast, GradScaler
from torch.optim import AdamW, lr_scheduler

from athena.checkpoint import save_checkpoint
from athena.dataloader import load_dataloader_pretrain
from athena.device import device
from athena.model import Athena, AthenaCompiled
from athena.utils import Throttle, Timer

class Pretrainer():
    
    def __init__(
        self,
        athena: Athena,
        optimizer: AdamW,
        scheduler: lr_scheduler,
        scaler: GradScaler,
        autocast_ctx: autocast,
        batch_size=4,
        backwards_every=5,
        log_every=0,
        save_every=120,
        valid_every=float("inf")
    ):
        self.athena = athena
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.athena_compiled = AthenaCompiled(self.athena)
        self.autocast_ctx = autocast_ctx
        
        self.batch_size = batch_size
        self.backwards_every = backwards_every
        self.save_every = save_every
        self.valid_every = valid_every
        self.log_every = log_every
        
        self._accumulation_counter = 0
        
    def train(self, epoch_limit, time_limit=float("inf")):
        self.athena.update_config(wandb_id=self.athena.wandb_id or wandb.util.generate_id())
        self.run = wandb.init(
            project="athena-pretrain",
            id=self.athena.wandb_id,
            name=self.athena.name,
            resume="allow",
            config={"batch_size": self.batch_size, **self.athena.config}
        )
        try:
            self.train_internal(epoch_limit, time_limit)
        except KeyboardInterrupt:
            pass
        wandb.finish()
        with Timer("Saving checkpoint"):
            save_checkpoint(self.athena, self.optimizer, self.scheduler, self.scaler)
        
    def train_internal(self, epoch_limit, time_limit):
        save_throttle = Throttle("Saving checkpoint", self.save_every)
        valid_throttle = Throttle("Validating", self.valid_every)
        log_throttle = Throttle(None, self.log_every)
            
        while True:
            train_dataloader, valid_dataloader = load_dataloader_pretrain(
                round(self.athena.context_size * self.athena.context_multiple) + 1,
                self.batch_size,
                resume_epoch=self.run.summary.get("epoch", 0)
            )
            
            total_steps_in_epoch = round(len(train_dataloader) / (1 - self.run.summary.get("epoch", 0) % 1))

            print("Starting epoch...")

            for batch in train_dataloader:
                step_info = self.step(batch, total_steps_in_epoch)
        
                with log_throttle as should_run:
                    if should_run:
                        print(
                            f"Step {step_info['step']} out of {total_steps_in_epoch} "
                            f"Time {step_info['training_time']:.4f} "
                            f"Loss {step_info['loss']:.4f} "
                            f"Step Time {step_info['step_time']:.4f} "
                            f"LR {step_info['lr']:.2e}"
                        )
                        wandb.log(step_info)

                with valid_throttle as should_run:
                    if should_run:
                        step_info["valid_loss"] = self.validate(valid_dataloader)
                        wandb.log({
                            "step": step_info["step"],
                            "valid_loss": step_info["valid_loss"]
                        })

                with save_throttle as should_run:
                    if should_run:
                        save_checkpoint(self.athena, self.optimizer, self.scheduler, self.scaler)
                        
                if self.run.summary.get("epoch", 0) >= epoch_limit or self.run.summary.get("training_time", 0) >= time_limit:
                    return
                
    def step(self, batch, steps_per_epoch):
        step_info = {}
        self.athena_compiled.train()

        with Timer() as step_timer:
     
            step_info["step"] = self.run.summary.get("step", 0) + 1

            with Timer("\tPrepping batch"):
                batch = batch.to(device, non_blocking=True)
                batch_x = batch[:, :-1].contiguous()
                batch_y = batch[:, 1:].contiguous()
            
            # Zero gradients only at start of accumulation cycle
            if self._accumulation_counter == 0:
                with Timer("\tZeroing gradients"):
                    self.optimizer.zero_grad(set_to_none=True)

            with Timer("\tForward pass"):
                with self.autocast_ctx:
                    output = self.athena_compiled(batch_x)["logits"]
                    loss = self.criterion(output, batch_y) / self.backwards_every

            with Timer("\tBackward pass"):
                step_info["loss"] = loss.detach().item() * self.backwards_every
                step_info["lr"] = self.optimizer.param_groups[0]['lr']
                self.scaler.scale(loss).backward()

            self._accumulation_counter += 1

            if self._accumulation_counter >= self.backwards_every:
                with Timer("\tStepping optimizer"):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self._accumulation_counter = 0
        
        self.scheduler.step()
        
        step_info["step_time"] = float(step_timer)
        
        # Hacky way to avoid messing up the graph when my computer falls asleep
        if float(step_timer) > 3 * self.run.summary.get("step_time", float("inf")):
            step_info["step_time"] = self.run.summary.get("step_time", float("inf"))
        
        step_info["training_time"] = self.run.summary.get("training_time", 0) + step_info["step_time"]
        step_info["epoch"] = step_info["step"] / steps_per_epoch

        self.run.summary["step"] = step_info["step"]
        self.run.summary["step_time"] = step_info["step_time"]
        self.run.summary["training_time"] = step_info["training_time"]
        self.run.summary["epoch"] = step_info["epoch"]

        return step_info

    def validate(self, valid_dataloader):
        self.athena_compiled.eval()
        valid_loss = 0.0
        for batch in valid_dataloader:
            batch = batch.to(device, non_blocking=True)
            batch_x = batch[:, :-1].contiguous()
            batch_y = batch[:, 1:].contiguous()
            with self.autocast_ctx:
                output = self.athena_compiled(batch_x)["logits"]
                valid_loss += self.criterion(output, batch_y).detach().item()
        valid_loss /= max(1, len(valid_dataloader))
        self.athena_compiled.train()
        return valid_loss

    def criterion(self, model_output, expected_output):
        logits = model_output.reshape(-1, model_output.shape[-1])
        labels = expected_output.reshape(-1)
        return functional.cross_entropy(logits, labels)
