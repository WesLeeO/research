import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class GPT2MC(nn.Module):
    def __init__(self, sft_model, tokenizer):
        super().__init__()
        self.model = sft_model
        self.tokenizer = tokenizer
        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.tokenizer.vocab_size
        self.q_function = nn.Linear(self.hidden_size, self.vocab_size)
        nn.init.zeros_(self.q_function.weight)
        nn.init.zeros_(self.q_function.bias)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, output_hidden_states=True, attention_mask=attention_mask)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        q_values = self.q_function(hidden_states) # [batch, seq_len, voc]
        return logits, q_values
