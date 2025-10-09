from environment import *
from torch.utils.data import Dataset, DataLoader
import torch

class QADataset2(Dataset):
    def __init__(self, conversations, tokenizer, max_conversations=20_000):
        self.data = []
        count = 0
        for convo in conversations:
            if max_conversations and count >= max_conversations:
                break

            texts = []
            for line in convo["lines"]:
                q, a = line.split("?", 1)
                texts.extend([Text("Q:" + q.strip() + "?", True), Text("A:" + a.strip(), False)])
            text_history: TextHistory = tuple(texts)
            reward = [0.0 if (not text.is_action or (convo['correct'] and i == len(convo["lines"]) - 1)) else -1.0 
                      for i, text in enumerate(texts)]
            done = True if convo['correct'] else convo['lines'][-1].startswith("Is the city")
            text_trajectory = TextTrajectory(text_history, tuple(reward), done)
            token_trajectory = TokenTrajectory.from_text_trajectory(text_trajectory, tokenizer)
            sample = self.truncate(token_trajectory, tokenizer.model_max_length, tokenizer.pad_token_id)
            self.data.append(sample)

    
    def truncate(self, token_trajectory: TokenTrajectory, max_len: int, pad_token_id: int) -> dict:

        input_ids = token_trajectory.tokens
        should_take_action = token_trajectory.is_action

        if len(input_ids) > max_len:
            start_idx = len(input_ids) - max_len
            input_ids = input_ids[start_idx:]
            should_take_action = should_take_action[start_idx:]

        attention_mask = np.ones_like(input_ids, dtype=np.int32)

        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = np.pad(input_ids, (0, pad_len), constant_values=pad_token_id)
            should_take_action = np.pad(should_take_action, (0, pad_len), constant_values=False)
            attention_mask = np.pad(attention_mask, (0, pad_len), constant_values=0)

        return {
            "input_ids": input_ids,
            "should_take_action": should_take_action,
            "attention_mask": attention_mask,
        }


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
