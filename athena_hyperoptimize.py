from athena.hyperoptimizer import AthenaHyperoptimizer
from argparse import ArgumentParser

argparser = ArgumentParser(description="Script for hyperoptimizing athena models")
argparser.add_argument("--init_model_config", type=dict, default={"embedding_size": 1024, "hidden_size": 2048, "num_layers": 1, "num_heads": 4, "head_size": 32, "key_size": 32, "vocab_size": 32768, "context_size": 256})
argparser.add_argument("--mutation_num", type=int, default=5)
argparser.add_argument("--mutation_rate", type=float, default=0.1)
argparser.add_argument("--mutation_time", type=float, default=1)
args = argparser.parse_args()

hyperoptimizer = AthenaHyperoptimizer(args.init_model_config, mutation_num=args.mutation_num, mutation_rate=args.mutation_rate, mutation_time=args.mutation_time)
hyperoptimizer.train(steps=1)

