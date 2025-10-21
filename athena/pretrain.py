import math
import re
import torch
import torch.nn.functional as functional
import wandb
from torch.amp import autocast, GradScaler
from torch.optim import AdamW

from athena.checkpoint import save_checkpoint
from athena.dataloader import load_dataloader_pretrain
from athena.device import device
from athena.model import Athena, AthenaCompiled
from athena.utils import Throttle, Timer, sample_indices_torch

from settings import pretrain_dataset_name, pretrain_dataset_train_chars, pretrain_dataset_valid_chars

class Pretrainer():
    
    def __init__(
        self,
        athena: Athena,
        optimizer: AdamW,
        scaler: GradScaler,
        autocast_ctx: autocast,
        train_batch_size=4,
        valid_batch_size=16,
        backwards_every=5,
        log_every=0,
        save_every=120,
        valid_every=float("inf")
    ):
        self.athena = athena
        self.optimizer = optimizer
        self.scaler = scaler
        self.athena_compiled = AthenaCompiled(self.athena)
        self.autocast_ctx = autocast_ctx
        
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
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
            config={"dataset": pretrain_dataset_name, "train_batch_size": self.train_batch_size, **self.athena.config}
        )
        try:
            self.train_internal(epoch_limit, time_limit)
        except KeyboardInterrupt:
            pass
        wandb.finish()
        with Timer("Saving checkpoint"):
            save_checkpoint(self.athena, self.optimizer, self.scaler)
        
    def train_internal(self, epoch_limit, time_limit):
        save_throttle = Throttle("Saving checkpoint", self.save_every)
        valid_throttle = Throttle("Validating", self.valid_every)
        log_throttle = Throttle(None, self.log_every)
            
        while True:
            train_dataloader, valid_dataloader = load_dataloader_pretrain(
                self.athena.context_size,
                self.train_batch_size,
                self.valid_batch_size,
                resume_chars=(self.run.summary.get("chars", 0) % pretrain_dataset_train_chars) % (pretrain_dataset_train_chars - 10000)
            )
            
            print("Starting epoch...")

            for char_index, batch in train_dataloader:
                
                assert char_index - pretrain_dataset_valid_chars < pretrain_dataset_train_chars
                
                step_info = {
                    "chars": (self.run.summary.get("chars", 0) // pretrain_dataset_train_chars) * pretrain_dataset_train_chars + (char_index - pretrain_dataset_valid_chars)
                }
                
                self.step(batch, step_info)
        
                with valid_throttle as should_run:
                    if should_run:
                        step_info["valid_loss"] = self.validate(valid_dataloader)
                        prev_best = self.run.summary.get("best_valid_loss", float("inf"))
                        if step_info["valid_loss"] < prev_best:
                            step_info["best_valid_loss"] = step_info["valid_loss"]
                            save_checkpoint(self.athena, self.optimizer, self.scaler, best=True)
                
                wandb.log(step_info)
                        
                with log_throttle as should_run:
                    if should_run:
                        print(
                            f"Chars {step_info['chars']} "
                            f"Step {step_info['step']} "
                            f"Epoch {step_info['epoch']} "
                            f"Time {step_info['training_time']:.4f} "
                            f"Loss {step_info['loss']:.4f} "
                            f"Step Time {step_info['step_time']:.4f} "
                        )

                with save_throttle as should_run:
                    if should_run:
                        save_checkpoint(self.athena, self.optimizer, self.scaler)
                        
                if self.run.summary.get("epoch", 0) >= epoch_limit or self.run.summary.get("training_time", 0) >= time_limit:
                    return
            
            wandb.log({"chars": round(self.run.summary.get("epoch", 0)) * pretrain_dataset_train_chars})
                
    def step(self, batch, step_info):
        self.athena_compiled.train()

        with Timer() as step_timer:
     
            step_info["epoch"] = step_info["chars"] / pretrain_dataset_train_chars
            step_info["step"] = self.run.summary.get("step", 0) + 1

            batch = batch.to(device, non_blocking=True)
            batch_x = batch[:, :-1].contiguous()
            batch_y = batch[:, 1:].contiguous()
        
            if self._accumulation_counter == 0:
                self.optimizer.zero_grad(set_to_none=True)

            with self.autocast_ctx:
                output = self.athena_compiled(batch_x)["logits"]
                loss = self.criterion(output, batch_y) / self.backwards_every

            step_info["loss"] = loss.detach().item() * self.backwards_every
            self.scaler.scale(loss).backward()

            self._accumulation_counter += 1

            if self._accumulation_counter >= self.backwards_every:
                self.scaler.step(self.optimizer, step_info["epoch"])
                self.scaler.update()
                self._accumulation_counter = 0
            
            for pg in self.optimizer.param_groups:
                
                for k, v in pg.items():
                    path = re.sub(r"\d+", "X", pg["name"]) + "." + k
                    if k in ["lr", "momentum", "update_lr"]:
                        step_info[path] = v
                
                weight: torch.Tensor = pg["params"][0].data
                
                if "monitored_indices" not in pg:
                    pg["monitored_indices"] = sample_indices_torch(weight.numel(), 5, pg["name"])
                    
                samples = zip(pg["monitored_indices"], weight.view(-1)[pg["monitored_indices"]].detach().cpu())
                
                step_info |= {f"{pg["name"]}.{i}": sample[1] for i, sample in enumerate(samples)}

        step_info["step_time"] = float(step_timer)
        
        # Hacky way to avoid messing up the graph when my computer falls asleep
        if float(step_timer) > 3 * self.run.summary.get("step_time", float("inf")):
            step_info["step_time"] = self.run.summary.get("step_time", float("inf"))
        
        step_info["training_time"] = self.run.summary.get("training_time", 0) + step_info["step_time"]

    def validate(self, valid_dataloader):
        self.athena_compiled.eval()
        valid_loss = 0.0
        num_batches = 0
        for i, batch in valid_dataloader:
            batch = batch.to(device, non_blocking=True)
            batch_x = batch[:, :-1].contiguous()
            batch_y = batch[:, 1:].contiguous()
            with self.autocast_ctx:
                output = self.athena_compiled(batch_x)["logits"]
                valid_loss += self.criterion(output, batch_y).detach().item()
            num_batches += 1
        valid_loss /= max(1, num_batches)
        self.athena_compiled.train()
        return valid_loss

    def criterion(self, model_output, expected_output):
        logits = model_output.reshape(-1, model_output.shape[-1])
        labels = expected_output.reshape(-1)
        return functional.cross_entropy(logits, labels)
