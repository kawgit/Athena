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

import torch
import os

from settings import *

def save_checkpoint(transformer, trainer):

    print(f"Saving checkpoint to {checkpoint_path}.")

    checkpoint = {
        "transformer": transformer.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "wandb_id": trainer.wandb_id,
        "epoch_idx": trainer.epoch_idx,
        "batch_idx": trainer.batch_idx,
    }

    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    
    if os.path.exists(checkpoint_path):
        os.rename(checkpoint_path, checkpoint_path + ".old")

    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(transformer, trainer):

    if resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}.")

        checkpoint = torch.load(checkpoint_path, weights_only=True)

        transformer.load_state_dict(checkpoint["transformer"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler"])
        trainer.wandb_id = checkpoint["wandb_id"]
        trainer.epoch_idx = checkpoint["epoch_idx"]
        trainer.batch_idx = checkpoint["batch_idx"]
