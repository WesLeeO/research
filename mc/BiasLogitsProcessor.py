from transformers import LogitsProcessor
import torch

class BiasLogitsProcessor(LogitsProcessor):
    def __init__(self, gpt2MC, alpha=-1.0):
        self.gpt2MC = gpt2MC
        self.alpha = alpha

    def __call__(self, input_ids, scores):
        with torch.no_grad():
            outputs = self.gpt2MC.model(input_ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :]  # (batch_size, hidden_dim)
            bias = self.gpt2MC.q_function(last_hidden)  # (batch_size, vocab_size)

        # Convert logits to probabilities
        probs = torch.softmax(scores, dim=-1)

        # Apply bias in probability space
        probs = probs ** (self.alpha * bias)

        # Renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

        # Convert back to logits for the generator
        scores = torch.log(probs + 1e-20)
        return scores
