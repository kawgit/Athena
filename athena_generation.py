import torch

from athena_tokenizer import end_token_id

class AthenaGeneration():
    def __init__(self, seed_tokens, reply_queue):

        self.cum_tokens = seed_tokens.copy()
        self.new_tokens = self.cum_tokens
        self.past_kvs = "init"

        self.replying = [False] * len(seed_tokens)
        self.reply_queue = [[]] * len(seed_tokens) if reply_queue == None else reply_queue
        self.active_batches = list(range(len(seed_tokens)))

    def push(self, new_tokens):

        self.new_tokens = new_tokens

        for batch_index, old_index in reversed(list(enumerate(self.active_batches))):

            batch_replying = self.replying[old_index]
            batch_new_token = self.new_tokens[batch_index]

            if batch_replying:

                self.write_reply(batch_index, batch_new_token)

            elif batch_new_token[0] == end_token_id:

                self.end_response(batch_index)
            
            self.cum_tokens[old_index].extend(batch_new_token)

    def write_reply(self, batch_index, batch_new_token):

        old_index = self.active_batches[batch_index]
        batch_reply_queue = self.reply_queue[old_index]

        if len(batch_reply_queue[0]) != 0:
            
            # Continuance of reply
            
            batch_new_token[0] = batch_reply_queue[0][0]
            del batch_reply_queue[0][0]

        else:
            
            # End of reply

            self.replying[old_index] = False
            del batch_reply_queue[0]
    
    def end_response(self, batch_index):

        old_index = self.active_batches[batch_index]

        if len(self.reply_queue[old_index]) != 0:

            # Start of reply

            self.replying[old_index] = True

        else:

            # End of batch's message chain

            del self.active_batches[batch_index]
            del self.new_tokens[batch_index]
            
            for i, (past_k, past_v) in enumerate(self.past_kvs):

                mask = torch.ones(past_k.shape[0], dtype=torch.bool)
                mask[batch_index] = False
                
                past_k = past_k[mask]
                past_v = past_v[mask]

                self.past_kvs[i] = (past_k, past_v)    
    
