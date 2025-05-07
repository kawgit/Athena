from tqdm import tqdm
import math
import numpy as np
import torch

from athena import load_athena
from athena_dataloader import load_dataloader
from athena_reward import calculate_reward, format_metric
from athena_tokenizer import tokenizer
from utils import parse_assistant_response

batch_size = 4
num_samples = 100
num_batches = math.ceil(num_samples / batch_size)

dataloader = load_dataloader(split="valid", batch_size=batch_size)
athena = load_athena()

num_batches = min(num_batches, len(dataloader))
total_metric = None

for i, entry in tqdm(enumerate(dataloader), total=num_batches):

    problems = entry['question']
    answers = entry['answer']

    problems = [f"<|user|>{problem}<|end|><|assistant|>" for problem in problems]
    problems = tokenizer(problems, padding=True, return_attention_mask=False)['input_ids']

    with torch.no_grad():
        histories = athena.generate(problems, 1000)
    
    rewards, metrics = zip(*[calculate_reward(history, answer) for history, answer in zip(histories, answers)])

    if total_metric is None:
        total_metric = np.zeros_like(metrics[0])

    for metric in metrics[1:]:
        total_metric += metric

    print(total_metric / ((i + 1) * batch_size))

    if i >= num_batches:
        break

print(format_metric(total_metric / (num_batches * batch_size)))