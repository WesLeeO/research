from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Any, Iterator
from transformers.tokenization_utils import PreTrainedTokenizer
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy

@dataclass(frozen=True)
class Text:
    text: str
    is_action: bool

TextHistory = Tuple[Text, ...]
text_history_to_str = lambda text_history: ''.join(map(lambda x: x.text, text_history))
StepResult = Tuple[TextHistory, float, bool]

# text trajectory should fit into a single context window, otherwise is truncated

@dataclass(frozen=True)
class TextTrajectory:
    text_history: TextHistory
    reward: Tuple[float, ...]
    done: bool

    def __post_init__(self):
        assert len(self.reward) == len(self.text_history), "reward is needed for each text"
        assert all([r == 0.0 for r, t in zip(self.reward, self.text_history) if not t.is_action]), "reward for non-actions texts should be 0.0"


@dataclass(frozen=True)
class TextTrajectoryChain:
    text_trajectory: TextTrajectory
    next: Optional[TextTrajectoryChain]



@dataclass(frozen=True)
class TokenHistory:
    tokens: np.ndarray # 1d int32 array
    is_action: np.ndarray # 1d bool array

    def __post_init__(self):
        assert len(self.tokens.shape) == 1 and len(self.is_action.shape) == 1, '(tokens, is_action) must be 1 dimensional'
        assert self.tokens.shape == self.is_action.shape, '(tokens, is_action) must have the same shape'
    
    @classmethod
    def from_text_history(
        cls, 
        text_history: TextHistory, 
        tokenizer: PreTrainedTokenizer, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> TokenHistory:
        if token_process is None:
            token_process = lambda x: x

        tokens = []
        is_action = []
        
        for item in text_history:
            
            # tokenize
            new_tokens = token_process(tokenizer.encode(item.text))
            
            tokens.extend(new_tokens)
            is_action.extend([item.is_action]*len(new_tokens))
        
        return cls(
            np.array(tokens, dtype=np.int32), 
            np.array(is_action, dtype=np.bool_), 
        )

@dataclass(frozen=True)
class TokenTrajectory:
    tokens: np.ndarray # 1d int32 array
    is_action: np.ndarray # 1d bool array
    reward: np.ndarray # 1d float32 array
    done: np.ndarray # bool scalar

    def __post_init__(self):
        assert len(self.tokens.shape) == 1, 'tokens must be 1 dimensional'
        assert len(self.is_action.shape) == 1, 'is_action must be 1 dimensional'
        assert len(self.reward.shape) == 1, 'reward must be 1 dimensional'
        assert len(self.done.shape) == 0, 'done must be scalar'

        assert self.is_action.shape == self.tokens.shape, 'is_action must have the same shape as tokens'
        assert self.reward.shape == self.tokens.shape, 'reward must have the same shape as tokens'

        assert not np.any(((1 - self.is_action.astype(np.float32)) * self.reward) != 0.0), 'reward must be 0.0 if not an action'
    
    @classmethod
    def from_text_trajectory(
        cls, 
        text_trajectory: TextTrajectory, 
        tokenizer: PreTrainedTokenizer, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> TokenTrajectory:
        if token_process is None:
            token_process = lambda x: x
        
        tokens = []
        is_action = []
        is_last_action_token = []
        reward = []

        for i, item in enumerate(text_trajectory.text_history):
            
            # tokenize
            new_tokens = token_process(tokenizer.encode(item.text))
            
            tokens.extend(new_tokens)
            is_action.extend([item.is_action]*len(new_tokens))            
            # add reward at the last token in the text
            reward.extend(([0.0]*(len(new_tokens)-1))+[text_trajectory.reward[i]])
        
        # get done
        done = text_trajectory.done

        return cls(
            np.array(tokens, dtype=np.int32), 
            np.array(is_action, dtype=np.bool_), 
            np.array(reward, dtype=np.float32), 
            np.array(done, dtype=np.bool_), 
        )

@dataclass(frozen=True)
class TokenTrajectoryChain:
    token_trajectory: TokenTrajectory
    next: Optional[TokenTrajectoryChain]

    def __post_init__(self):
        curr, dones = self, []
        while curr.next is not None:
            dones.append(curr.token_trajectory.done)
            curr = curr.next
        assert not np.any(dones[:-1]), 'token trajectory chain can only be done at the end'
    
    def to_list(self) -> List[TokenTrajectory]:
        curr, l = self, []
        while curr is not None:
            l.append(curr.token_trajectory)
            curr = curr.next
        return l

    @classmethod
    def from_text_trajectory_chain(
        cls, 
        text_trajectory_chain: TextTrajectoryChain, 
        tokenizer: PreTrainedTokenizer, 
        token_process: Optional[Callable[[List[int]], List[int]]]=None, 
    ) -> TokenTrajectoryChain:
        return TokenTrajectoryChain(
            TokenTrajectory.from_text_trajectory(
                text_trajectory_chain.text_trajectory, 
                tokenizer, 
                token_process=token_process, 
            ), 
            cls.from_text_trajectory_chain(
                text_trajectory_chain.next, 
                tokenizer, 
                token_process=token_process, 
            ) if text_trajectory_chain.next is not None else None, 
        )