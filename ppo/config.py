
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Main configuration for training."""
    
    # Model paths
    base_model_path: str = "./gpt2-medium-offline"
    sft_model_path: str = "bc/fine_tuned_gpt2_medium_lora_filtered"
    rl_output_dir: str = "ppo/rl_models"
    
    # Oracle configuration
    oracle_model: str = "microsoft/Phi-3-mini-4k-instruct"
    
    # Game settings
    max_questions: int = 20
    
    # PPO hyperparameters
    rl_learning_rate: float = 1e-4
    rl_batch_size: int = 8
    rl_mini_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5
    
    # PPO-specific
    init_kl_coef: float = 0.2
    target_kl: float = 6.0
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95   # GAE lambda
    
    # Generation parameters
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
    
    # Reward parameters
    question_penalty: float = -0.1
    correct_guess_reward: float = 10.0
    wrong_guess_penalty: float = -5.0
    efficiency_bonus_weight: float = 5.0
    
    # Training schedule
    episodes_per_iteration: int = 8
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100
    
    """
    # Logging
    log_with: str = "wandb"  # "wandb", "tensorboard", or None
    wandb_project: str = "city-guesser-rl"
    
    # Misc
    seed: int = 42
    """