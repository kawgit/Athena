import torch
import torch.nn.functional as functional

from athena.device import device

class AthenaGeneration():
    def __init__(self, model, model_compiled, seed_tokens, temperature=1.0):

        self.model = model
        self.model_compiled = model_compiled if model_compiled != None else model
        self.temperature = temperature
        self.batch_size = len(seed_tokens)

        self.cum_tokens = torch.tensor(seed_tokens, dtype=torch.long) if type(seed_tokens) == list else seed_tokens.cpu()
        self.cum_logits = [0] * self.batch_size
        self.cum_grads = [None] * self.batch_size

        self.new_tokens = self.cum_tokens.to(device)
        self.past_kvs = "init"

    def step(self):
        
        use_cache = self.past_kvs != None and self.past_kvs != "init"
        
        if use_cache and self.past_kvs[0].shape[3] >= self.model.context_size:
            past_ks, past_vs = self.past_kvs
            past_ks_truncated = past_ks[:, :, :, -self.model.context_size+1:]
            past_vs_truncated = past_vs[:, :, :, -self.model.context_size+1:]
            self.past_kvs = (past_ks_truncated, past_vs_truncated)
        
        results = self.model_compiled(self.new_tokens, past_kvs=self.past_kvs)
        logits = results["logits"][:, -1, :] / self.temperature
        self.past_kvs = results["present_kvs"]
    
        probs = functional.softmax(logits, dim=-1)
        self.new_tokens = torch.multinomial(probs, num_samples=1, replacement=True)
        self.cum_tokens = torch.cat((self.cum_tokens, self.new_tokens.cpu()), dim=-1)

        for batch_index, new_token in enumerate(self.new_tokens.flatten()):

            logit = logits[batch_index][new_token]
            self.cum_logits[batch_index] += logit.detach().item()

            if torch.is_grad_enabled():
                
                new_grad = torch.autograd.grad(logit, self.model.parameters(), retain_graph=batch_index < self.batch_size - 1, allow_unused=True)
                
                if self.cum_grads[batch_index] != None:
                    self.cum_grads[batch_index] = [g1 + g2 if g1 != None and g2 != None else g1 or g2 for g1, g2 in zip(self.cum_grads[batch_index], new_grad)]
                else:
                    self.cum_grads[batch_index] = new_grad