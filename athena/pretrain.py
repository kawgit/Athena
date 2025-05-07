import torch.nn.functional as functional
import wandb
from torch.amp import GradScaler, autocast

from athena.checkpoint import save_checkpoint
from athena.dataloader import load_dataloader_pretrain
from athena.device import device
from athena.model import AthenaCompiled
from athena.utils import Throttle, Timer

def pretrain(athena, optimizer, scheduler, batch_size=4, save_every=120, valid_every=30, epoch_limit=1, time_limit=float("inf")):

    athena_compiled = AthenaCompiled(athena)
    train_dataloader, valid_dataloader = load_dataloader_pretrain(athena.config.context_size + 1, batch_size)
    scaler = GradScaler(device.type)
    
    resume = athena.config.wandb_id is not None
    athena.config.wandb_id = athena.config.wandb_id or wandb.util.generate_id()
    run = wandb.init(
        project="athena-pretrain",
        id=athena.config.wandb_id,
        name=athena.config.name,
        resume="must" if resume else None,
        config={
            **vars(athena.config),
            "batch_size": batch_size,
        }
    )
    
    last_step_info = {"step": 0, "training_time": 0, "step_time": float("inf")}
    
    if resume:
        api = wandb.Api()
        previous_run = api.run(f"{run.entity}/{run.project}/{run.id}")
        history = previous_run.history(keys=["step", "training_time", "step_time"])
        last_row = history.iloc[-1] if not history.empty else {}
        last_step_info["step"] = int(last_row["step"])
        last_step_info["training_time"] = float(last_row["training_time"])
        last_step_info["step_time"] = float(last_row["step_time"])

    save_throttle = Throttle(f"Saving checkpoint", save_every)
    valid_throttle = Throttle(f"Validating", valid_every)
    
    try:
        for epoch in range(epoch_limit):
            for i, batch in enumerate(train_dataloader):
                
                step_info = pretrain_step(athena_compiled, optimizer, train_dataloader, scaler, last_step_info, batch)
                
                with valid_throttle as should_run:
                    if should_run:
                        step_info["valid_loss"] = pretrain_validate(athena_compiled, valid_dataloader)
                        scheduler.step(step_info["valid_loss"])

                with save_throttle as should_run:
                    if should_run:
                        save_checkpoint(athena, optimizer, scheduler)
                
                wandb.log(step_info)
                last_step_info = step_info
                
                if last_step_info["training_time"] > time_limit:
                    break
    except KeyboardInterrupt:
        pass
    
    if "valid_loss" not in step_info:
        with Timer("Final validation"):
            step_info["valid_loss"] = pretrain_validate(athena_compiled, valid_dataloader)

    wandb.finish()

    with Timer("Saving checkpoint"):
        save_checkpoint(athena, optimizer, scheduler)
        
def pretrain_step(athena_compiled, optimizer, train_dataloader, scaler, last_step_info, batch):
    
    step_info = {}
    athena_compiled.train()

    with Timer() as step_timer:
        
        step_info["step"] = last_step_info["step"] + 1
        step_info["epoch"] = step_info["step"] / len(train_dataloader)
        
        batch = batch.to(device, non_blocking=True)
        batch_x = batch[:, :-1]
        batch_y = batch[:, 1:]
        
        optimizer.zero_grad()

        with autocast(device.type):
            output = athena_compiled(batch_x)["logits"]
            loss = pretrain_criterion(output, batch_y)

        step_info["loss"] = loss.detach().item()
        step_info["lr"] = optimizer.param_groups[0]['lr']
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    step_info["step_time"] = float(step_timer) if float(step_timer) < 10 * last_step_info["step_time"] else last_step_info["step_time"]
    step_info["training_time"] = last_step_info["training_time"] + step_info["step_time"]
    
    print(f"Step {step_info['step']} out of {len(train_dataloader)} Time {step_info['training_time']:.4f} Loss {step_info['loss']:.4f} Step Time {step_info['step_time']:.4f} LR {step_info['lr']:.4f}")
    
    return step_info

def pretrain_validate(athena_compiled, valid_dataloader):

    athena_compiled.eval()
    valid_loss = 0
    for batch in valid_dataloader:
        batch = batch.to(device, non_blocking=True)
        batch_x = batch[:, :-1]
        batch_y = batch[:, 1:]
        
        with autocast(device.type):
            output = athena_compiled(batch_x)["logits"]
            valid_loss += pretrain_criterion(output, batch_y).detach().item()
            
    valid_loss /= len(valid_dataloader)
    athena_compiled.train()
    
    return valid_loss

def pretrain_criterion(model_output, expected_output):
    
    logits = model_output.reshape(-1, model_output.shape[-1])
    labels = expected_output.reshape(-1)
    
    return functional.cross_entropy(logits, labels)