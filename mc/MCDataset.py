from environment import *
from data import *
from torch.utils.data import Dataset, DataLoader
import torch

class MCDataset(Dataset):
    def __init__(self, conversations, tokenizer, gamma=0.99):
        self.data = []
        count = 0
        for convo in conversations:
            if convo['word'].split(',')[0].lower() in convo['lines'][-1].split('?')[0].lower(): #BC-filtered:
                texts = []
                for line in convo["lines"]:
                    q, a = line.split("?", 1)
                    texts.extend([Text("Q:" + q.strip() + "?", True), Text("A:" + a.strip(), False)])
                text_history: TextHistory = tuple(texts)
                reward = [0.0 if (not text.is_action or (convo['correct'] and i == len(convo["lines"]) - 1)) else -1.0 
                        for i, text in enumerate(texts)]
                done = True if convo['correct'] else convo['lines'][-1].startswith("Is the city")
                text_trajectory = TextTrajectory(text_history, tuple(reward), done)
                text_trajectory_chain = TextTrajectoryChain(text_trajectory, None)
                token_trajectory_chain = TokenTrajectoryChain.from_text_trajectory_chain(text_trajectory_chain, tokenizer)
                mc_data = self.truncate_mc_data(MCData.from_token_trajectory_chain(token_trajectory_chain, gamma=gamma), tokenizer.model_max_length, tokenizer.pad_token_id)
                self.data.append(mc_data)
                count += 1

    def truncate_mc_data(self, mc_data: MCData, max_len: int, pad_token_id: int) -> dict:
        input_ids = mc_data.input_ids
        should_take_action = mc_data.should_take_action
        returns = mc_data.returns

        # Truncate left if needed
        if len(input_ids) > max_len:
            start_idx = len(input_ids) - max_len
            input_ids = input_ids[start_idx:]
            should_take_action = should_take_action[start_idx:]
            returns = returns[start_idx:]

        # Create attention mask before padding
        attention_mask = np.ones_like(input_ids, dtype=np.int32)

        # Pad right if needed
        pad_len = max_len - len(input_ids)
        if pad_len > 0:
            input_ids = np.pad(input_ids, (0, pad_len), constant_values=pad_token_id)
            should_take_action = np.pad(should_take_action, (0, pad_len), constant_values=False)
            returns = np.pad(returns, (0, pad_len), constant_values=0.0)
            attention_mask = np.pad(attention_mask, (0, pad_len), constant_values=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "should_take_action": should_take_action,
            "returns": returns
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
