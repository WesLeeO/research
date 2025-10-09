from __future__ import annotations
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import NamedTuple, List, Dict
from transformers import PreTrainedTokenizerBase
from torch.utils.data import IterableDataset


def get_rtg(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute discounted reward-to-go (RTG) sequence.
    rewards: [T] numpy array of rewards
    gamma: discount factor
    returns: [T] numpy array of RTG values
    """
    T = rewards.shape[0]
    rtg = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for t in reversed(range(T)):
        running_return = rewards[t] + gamma * running_return
        rtg[t] = running_return
    return rtg

class MCData(NamedTuple):
    input_ids: np.ndarray      # [t]
    should_take_action: np.ndarray  # [t-1] bool mask
    returns: np.ndarray        # [t-1]

    @staticmethod
    def block(
        data: List[MCData],
        blocking_strategy,  # user-defined (padding/truncation)
        tokenizer: PreTrainedTokenizerBase,
    ) -> Dict[str, np.ndarray]:
        """
        Collate a list of MCData objects into padded numpy arrays.
        """
        def block_sequences(seq_list, pad_value, dtype, max_length):
            # Simple version of block_sequences: pad to max_length
            max_len = min(max(len(s) for s in seq_list), max_length)
            out = np.full((len(seq_list), max_len), pad_value, dtype=dtype)
            for i, seq in enumerate(seq_list):
                trunc = seq[:max_len]
                out[i, :len(trunc)] = trunc
            return out

        max_len = blocking_strategy.max_length

        return dict(
            input_ids=block_sequences(
                [x.input_ids for x in data],
                tokenizer.pad_token_id,
                np.int32,
                max_len,
            ),
            should_take_action=block_sequences(
                [x.should_take_action for x in data],
                False,
                np.bool_,
                max_len - 1,
            ),
            returns=block_sequences(
                [x.returns for x in data],
                0.0,
                np.float32,
                max_len - 1,
            ),
        )

    @classmethod
    def from_token_trajectory_chain(
        cls,
        token_trajectory_chain,
        gamma: float,
    ):
        # Flatten rewards across the chain
        all_tokens = []
        all_is_action = []
        all_rewards = []

        # collect all trajectories
        for token_traj in token_trajectory_chain.to_list():
            all_tokens.append(token_traj.tokens)
            all_is_action.append(token_traj.is_action)
            all_rewards.append(token_traj.reward)

        # concatenate full conversation
        input_ids = np.concatenate(all_tokens, axis=0)
        is_action = np.concatenate(all_is_action, axis=0)
        rewards = np.concatenate(all_rewards, axis=0)

        # compute RTG only for action tokens
        filtered_rewards = rewards[is_action]
        rtgs_sequence = get_rtg(filtered_rewards, gamma=gamma)
        
        # assign RTG back to action tokens
        returns = np.zeros_like(is_action, dtype=np.float32)
        returns[is_action] = rtgs_sequence[:is_action.sum()]

        return cls(
            input_ids=input_ids,
            should_take_action=is_action,
            returns=returns
        )
        """
        filtered_rewards_chain = []
        should_take_action_chain = []

        for token_trajectory in token_trajectory_chain.to_list():
            should_take_action = token_trajectory.is_action[1:]
            rewards = token_trajectory.reward[1:]
            filtered_rewards = rewards[should_take_action]

            filtered_rewards_chain.append(filtered_rewards)
            should_take_action_chain.append(should_take_action)

        filtered_rewards_chain = np.concatenate(filtered_rewards_chain, axis=0)
        should_take_action_chain = np.concatenate(should_take_action_chain, axis=0)

        rtgs_sequence = get_rtg(filtered_rewards_chain, gamma=gamma)

        should_take_action = token_trajectory_chain.token_trajectory.is_action[1:]
        returns = np.zeros_like(should_take_action, dtype=np.float32)
        returns[should_take_action] = rtgs_sequence[:should_take_action.sum()]

        return cls(
            input_ids=token_trajectory_chain.token_trajectory.tokens,
            should_take_action=should_take_action,
            returns=returns,
        )
        """


class MCDataset(Dataset):
    def __init__(
        self,
        input_ids: np.ndarray,         # [b, t]
        should_take_action: np.ndarray,  # [b, t-1]
        returns: np.ndarray,           # [b, t-1]
    ):
        assert input_ids.shape[1] == should_take_action.shape[1] + 1
        assert input_ids.shape[1] == returns.shape[1] + 1

        assert input_ids.shape[0] == should_take_action.shape[0]
        assert input_ids.shape[0] == returns.shape[0]

        self.input_ids = input_ids
        self.should_take_action = should_take_action
        self.returns = returns

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
            "should_take_action": torch.tensor(self.should_take_action[index], dtype=torch.bool),
            "returns": torch.tensor(self.returns[index], dtype=torch.float32),
        }

    def __len__(self):
        return self.input_ids.shape[0]

    @classmethod
    def from_mc_data_list(
        cls,
        mc_data_list: List[MCData],
        tokenizer: PreTrainedTokenizerBase,
        blocking_strategy,
    ):
        data = MCData.block(mc_data_list, blocking_strategy, tokenizer)
        return cls(**data)


class _MCIteratorDataset:
    def __init__(self, mc_data_iterator):
        self.mc_data_iterator = mc_data_iterator

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.mc_data_iterator)
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'should_take_action': torch.tensor(item['should_take_action'], dtype=torch.bool),
            'returns': torch.tensor(item['returns'], dtype=torch.float),
        }

class MCIterableDataset(IterableDataset):
    def __init__(self, mc_data_iterable):
        self.mc_data_iterable = mc_data_iterable
    
    def __iter__(self):
        return _MCIteratorDataset(iter(self.mc_data_iterable))
    
    @classmethod
    def from_mc_data_iterable(
        cls, 
        mc_data: Iterable[MCData], 
        tokenizer, 
        blocking_strategy,
    ) -> MCIterableDataset:

        class _TokensIterable:
            def __iter__(self):
                for item in mc_data:
                    blocked = MCData.block([item], blocking_strategy, tokenizer)
                    # take the first (and only) element after blocking
                    yield {
                        'input_ids': blocked['input_ids'][0],
                        'should_take_action': blocked['should_take_action'][0],
                        'returns': blocked['returns'][0],
                    }

        return cls(_TokensIterable())
