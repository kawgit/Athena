import random
import time
import torch
import torch.nn.functional as functional
import wandb
from torch.amp import GradScaler

from athena.checkpoint import load_checkpoint, save_checkpoint
from athena.dataloader import load_dataloader_pretrain
from athena.device import device
from athena.model import AthenaCompiled
from athena.utils import Timer, left_pad_and_stack
from settings import checkpoint_path

batch_size = 1
time_between_saves = 120
accumulation_steps = 1

athena, optimizer = load_checkpoint(checkpoint_path)
athena_compiled = AthenaCompiled(athena)
dataloader = load_dataloader_pretrain(athena.config.context_size, batch_size=batch_size)

resume_mode = "allow" if athena.config.wandb_id is not None else None
athena.config.wandb_id = athena.config.wandb_id or wandb.util.generate_id()

wandb.init(
    project="athena-pretrain-ppo",
    id=athena.config.wandb_id,
    resume=resume_mode,
    config={
        **vars(athena.config),
        "batch_size": batch_size,
        "time_between_saves": time_between_saves,
    }
)

time_of_last_save = time.time()
start = time.time()
scaler = GradScaler(device.type)
quality_rolling_average = 0
quality_rolling_average_decay = 0.5

try: 
    for i, batch in enumerate(dataloader):

        batch = batch.to(device, non_blocking=True)
        
        seed_len = random.randint(1, round(athena.config.context_size) - 1)
        gen_len = athena.config.context_size - seed_len
        
        batch_x = batch[:, :seed_len]
        batch_y = batch
        
        expected_qualities = torch.zeros_like(batch_y, device=device, dtype=torch.float32)
        expected_qualities[:, :seed_len] = 1

        with Timer("\tGenerating responses"):

            with torch.autocast(device.type):
                generation = athena_compiled.generate(batch_x, gen_len)
        
        athena.zero_grad()
        
        with Timer("\tGenerating response quality loss"):
                
            with torch.autocast(device.type):
                qualities = athena_compiled(left_pad_and_stack(generation.cum_tokens).to(device), return_logits=False, return_qualities=True)["qualities"]
                quality_loss = functional.binary_cross_entropy_with_logits(qualities, expected_qualities).mean()
                

        with Timer("\tStepping quality optimizer"):

            scaler.scale(quality_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        with Timer("\tStepping response optimizer"):
            
            qualities = qualities.detach()[:, -1]
            quality_rolling_average = quality_rolling_average * quality_rolling_average_decay + qualities.mean() * (1 - quality_rolling_average_decay)
            
            advantages = (qualities - quality_rolling_average) / batch_size

            for param, gs in zip(athena.parameters(), zip(*generation.cum_grads)):
                nontrivial = [g for g in gs if g != None]
                if len(nontrivial) == 0:
                    continue
                
                param.grad = torch.stack(nontrivial).sum(dim=0)

            scaler.step(optimizer)
            scaler.update()

        print(f"Batch {i} Quality Loss {quality_loss.detach().item():.4f} Time {(time.time() - start):.4f} Tokens {gen_len} TPS {batch_size * gen_len / (time.time() - start)} Response Qualities {qualities.cpu().tolist()} Rolling Average {quality_rolling_average:.4f}")
        wandb.log({"quality_loss": quality_loss.detach().item(), "lr": optimizer.param_groups[0]['lr']})

        if time.time() - time_of_last_save > time_between_saves:
            with Timer(f"Saving checkpoint"):
                save_checkpoint(checkpoint_path, athena, optimizer)
            time_of_last_save = time.time()

        start = time.time()

except KeyboardInterrupt:
    pass

print(f"Saving checkpoint... ", end='')
start = time.time()
save_checkpoint(checkpoint_path, athena, optimizer)
time_of_last_save = time.time()
print(f"Done in {time.time() - start:.4f} seconds.")
wandb.finish()
