from typing import List, Dict
import numpy as np
from config import TrainingConfig


class RewardShaper:
    """
    Handles reward computation and shaping for sparse reward setting.
    
    In sparse rewards, we need special handling:
    1. Reward normalization
    2. Advantage estimation (PPO handles this)
    3. Optional reward shaping (credit assignment)
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def compute_question_reward(self, question_num: int) -> float:
        """
        Reward for asking a question.
        In sparse setting: always -1
        """
        return self.config.question_penalty  # -1.0
    
    def compute_guess_reward(
        self, 
        correct: bool, 
        questions_used: int
    ) -> float:
        """
        Reward for making final guess.
        
        Args:
            correct: Whether guess was correct
            questions_used: Number of questions before guess
            
        Returns:
            Final reward value
        """
        if correct:
            # Correct guess: big positive reward
            base_reward = self.config.correct_guess_reward  # 10.0
            
            # Efficiency bonus: fewer questions = more bonus
            efficiency = (self.config.max_questions - questions_used) / self.config.max_questions
            efficiency_bonus = self.config.efficiency_bonus_weight * efficiency
            
            total_reward = base_reward + efficiency_bonus
            
            return total_reward
        else:
            # Wrong guess: penalty
            return self.config.wrong_guess_penalty  # -5.0
    
    def apply_reward_to_go(self, rewards: List[float]) -> List[float]:
        """
        Apply reward-to-go for sparse rewards.
        This helps with credit assignment.
        
        Example episode:
        Actions:  [Q1,    Q2,    Q3,    Guess]
        Rewards:  [-1,    -1,    -1,    +10]
        
        Reward-to-go:
        R2G:      [+7,    +8,    +9,    +10]
        
        This helps model understand that early questions 
        were valuable even though they got -1.
        """
        rewards_to_go = []
        cumulative = 0
        
        # Go backwards through rewards
        for r in reversed(rewards):
            cumulative += r
            rewards_to_go.insert(0, cumulative)
        
        return rewards_to_go
    
    def apply_discount(
        self, 
        rewards: List[float],
        gamma: float = 0.99
    ) -> List[float]:
        """
        Apply discount factor to rewards.
        
        Discounted reward-to-go:
        R2G[t] = r[t] + gamma * r[t+1] + gamma^2 * r[t+2] + ...
        """
        discounted = []
        cumulative = 0
        
        for r in reversed(rewards):
            cumulative = r + gamma * cumulative
            discounted.insert(0, cumulative)
        
        return discounted