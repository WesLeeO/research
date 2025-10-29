
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Main configuration for training."""
    
    # Model paths
    base_model_path: str = "./gpt2-medium-offline"
    sft_model_path: str = "bc/fine_tuned_gpt2_medium_lora_filtered"
    rl_output_dir: str = "ppo/rl_models"
    
    max_context_length: str = 1024

    # Oracle configuration
    oracle_model: str = "microsoft/Phi-3-mini-4k-instruct"
    
    # Game settings
    max_questions: int = 20
    
    
    # PPO hyperparameters
    rl_learning_rate: float = 1e-4
    value_learning_rate: float = 1e-4
    rl_batch_size: int = 8
    rl_mini_batch_size: int = 2
    num_ppo_epochs: int = 4
    max_grad_norm: float = 0.5
    vf_coef: float = 0.5

    # PPO-specific
    kl_coef: float = 0.05
    target_kl: float = 6.0
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95   # GAE lambda
    value_clip_epsilon: float = 0.2
    cliprange : float = 0.2

    # Generation parameters
    max_new_tokens_question: int = 30
    max_new_tokens_answer: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95
        
    # Training schedule
    episodes_per_iteration: int = 8
    log_every: int = 10
    eval_every: int = 50
    #save_every: int = 100

    project: str = "research-multi-turn-RL-with-LMs"
    name: str ="gpt2-medium-ppo"
    log_with: str = "wandb"


    
    """
    # Logging
    log_with: str = "wandb"  # "wandb", "tensorboard", or None
    wandb_project: str = "city-guesser-rl"
    
    # Misc
    seed: int = 42
    """