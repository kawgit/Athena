import argparse
import wandb
from wandb import Api

parser = argparse.ArgumentParser()
parser.add_argument("--entity",       default="kawgit56-none")
parser.add_argument("--project",      default="athena-pretrain")
parser.add_argument("--base_run_id",  required=True)
parser.add_argument("--new_run_name", required=True)
args = parser.parse_args()

wandb.login()
api = Api()
base_run = api.run(f"{args.entity}/{args.project}/{args.base_run_id}")
history = base_run.history(samples=base_run.lastHistoryStep + 1)
files   = base_run.files()

run = wandb.init(
    project=args.project,
    entity=args.entity,
    config=base_run.config,
    name=args.new_run_name,
    resume="allow"
)

step_size = history["_step"].values[1] if "_step" in history else 1
for idx, row in history.iterrows():
    run.log(row.to_dict(), step=idx * step_size)

for f in files:
    f.download(replace=True)
    run.save(f.name, policy="now")

run.finish()
