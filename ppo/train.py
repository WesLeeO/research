import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from typing import List, Tuple
import wandb
from tqdm import tqdm

# Your custom modules

from environment import CityGuessingEnvironment, LLMOracle
from data_utils import CityDataset
from reward_shaper import RewardShaper
from config import TrainingConfig
from evaluation import Evaluator


class CityGuesserRLTrainer:
    """Main trainer class for RL fine-tuning with TRL."""
    
    def __init__(
        self,
        base_model_path: str
        sft_model_path: str,
        cities_data: List[dict],
        config: TrainingConfig
    ):
        self.config = config
        self.cities_data = cities_data
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load SFT model with value head for PPO
        print("Loading SFT model...")

        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True)
        sft_model = PeftModel.from_pretrained(base_model, sft_model_path)
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(sft_model)

        #does it need head really?      
        # Reference model (frozen copy of SFT model for KL penalty)
        self.ref_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True), sft_model_path)
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.ref_model.to(self.device)
        
        # Initialize environment with oracle
        oracle = LLMOracle(model_name=config.oracle_model)
        self.env = CityGuessingEnvironment(
            cities_data=cities_data,
            oracle=oracle,
            max_questions=config.max_questions
        )
        
        # Reward shaper
        self.reward_shaper = RewardShaper(config)
        
        # PPO configuration
        ppo_config = PPOConfig(
            model_name=sft_model_path,
            learning_rate=config.rl_learning_rate,
            batch_size=config.rl_batch_size,
            mini_batch_size=config.rl_mini_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            ppo_epochs=config.ppo_epochs,
            max_grad_norm=config.max_grad_norm,
            seed=config.seed,
            init_kl_coef=config.init_kl_coef,
            target=config.target_kl,
            gamma=config.gamma,  # Discount factor
            lam=config.lam,      # GAE lambda
            log_with=config.log_with,  # "wandb", "tensorboard", or None
        )
        
        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )
        
        # Generation parameters
        self.generation_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "do_sample": True,
            "top_p": config.top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Evaluator
        self.evaluator = Evaluator(self.env, self.tokenizer, self.generation_kwargs)
    
    def create_prompt(self, state: dict) -> str:
        """
        Create prompt from game state.
        """
        
        prompt = ""

        # Add conversation history
        for entry in state["history"]:
            if entry["type"] == "question":
                prompt += f"Q: {entry['content']}\nA: {entry['answer']}\n"
    
        prompt += "Q: "

        return prompt
    
    def play_episode(self) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[float], dict]:
        """
        Play one complete episode (game) and collect rollout data.
        
        Returns:
            query_tensors: List of tokenized prompts
            response_tensors: List of tokenized responses
            rewards: List of rewards for each step
            info: Episode summary info
        """
        state = self.env.reset()
        
        context_tensors = []
        question_tensors = []
        rewards = []
        
        done = False
        total_reward = 0
        
        while not done:
            # Create prompt from current state
            prompt = self.create_prompt(state)
            
            # Tokenize prompt
            context_tensor = self.tokenizer.encode(
                prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=self.config.max_context_length,
                truncation_side="left" 
            ).squeeze().to(self.device)
            
            # Generate response using current policy
            with torch.no_grad():
                question_tensor = self.ppo_trainer.generate(
                    context_tensor.unsqueeze(0),
                    return_prompt=False,  # Only return generated part
                    **self.generation_kwargs
                ).squeeze()
            
            # Decode response
            next_question = self.tokenizer.decode(
                question_tensor,
                skip_special_tokens=True
            ).strip()

            next_question = (next_question.split("?")[0] + "?").strip()
            # Take action in environment
            state, reward, done, info = self.env.step(next_question)
            
            # Store rollout data
            context_tensors.append(context_tensor)
            question_tensors.append(question_tensor)
            rewards.append(reward)
            
            total_reward += reward
            
            # Safety: prevent infinite loops
            """
            if len(query_tensors) > 30:
                # Force end with penalty
                done = True
                rewards[-1] = -10.0
                info["timeout"] = True
            """
        
        # Episode info for logging
        episode_info = {
            "total_reward": total_reward,
            "correct": info.get("correct", False),
            "questions_used": info.get("questions_used", len(question_tensors)),
            "target_city": info.get("target_city", "unknown"),
            "efficiency": info.get("efficiency", 0.0),
            "timeout": info.get("timeout", False)
        }
        
        return context_tensors, question_tensors, rewards, episode_info
    
    def collect_batch(self, num_episodes: int) -> Tuple[List, List, List, List]:
        """
        Collect a batch of episodes for PPO update.
        
        Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            All queries, responses, rewards, and episode info
        """
        all_contexts = []
        all_questions = []
        all_rewards = []
        all_infos = []
        
        for _ in range(num_episodes):
            contexts, questions, rewards, info = self.play_episode()
            all_contexts.extend(contexts)
            all_questions.extend(questions)
            all_rewards.extend(rewards)
            all_infos.append(info)
        
        return all_contexts, all_questions, all_rewards, all_infos
    
    def train(self, num_iterations: int, episodes_per_iteration: int = 8):
        """
        Main training loop.
        
        Args:
            num_iterations: Total number of PPO updates
            episodes_per_iteration: Episodes to collect before each update
        """
        print(f"Starting RL training for {num_iterations} iterations...")
        print(f"Collecting {episodes_per_iteration} episodes per iteration")
        
        for iteration in tqdm(range(num_iterations)):
            # Collect batch of episodes
            all_contexts, all_questions, all_rewards, all_infos = self.collect_batch(
                episodes_per_iteration
            )
            
            # Convert rewards to tensors
            reward_tensors = [torch.tensor(r, dtype=torch.float32) for r in all_rewards]
            
            # PPO update step - THIS IS WHERE TRL DOES THE MAGIC
            stats = self.ppo_trainer.step(
                all_contexts,
                all_questions,
                reward_tensors
            )
            
            # Logging
            if iteration % self.config.log_every == 0:
                self._log_training_stats(iteration, infos, stats)
            
            # Evaluation
            if iteration % self.config.eval_every == 0:
                eval_results = self.evaluator.evaluate(
                    self.model, 
                    num_episodes=20
                )
                self._log_eval_stats(iteration, eval_results)
            
            # Save checkpoint
          #  if iteration % self.config.save_every == 0:
          #      self.save_checkpoint(f"checkpoint_iter_{iteration}")
        
        # Final save
        self.save_checkpoint("ppo_model")
        print("Training complete!")
    
    def _log_training_stats(self, iteration: int, infos: List[dict], stats: dict):
        """Log training statistics."""
        # Episode-level metrics
        success_rate = sum(info["correct"] for info in infos) / len(infos)
        avg_questions = sum(info["questions_used"] for info in infos) / len(infos)
        avg_reward = sum(info["total_reward"] for info in infos) / len(infos)
        avg_efficiency = sum(info["efficiency"] for info in infos) / len(infos)
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Avg Questions: {avg_questions:.1f}")
        print(f"Avg Reward: {avg_reward:.2f}")
        print(f"Avg Efficiency: {avg_efficiency:.2%}")
        
        # PPO stats from TRL
        if stats:
            print(f"\nPPO Stats:")
            print(f"  Policy Loss: {stats.get('ppo/loss/policy', 0):.4f}")
            print(f"  Value Loss: {stats.get('ppo/loss/value', 0):.4f}")
            print(f"  KL Divergence: {stats.get('ppo/mean_non_score_reward', 0):.4f}")
        print(f"{'='*60}\n")
        
        # Log to wandb if enabled
        if self.config.log_with == "wandb":
            wandb.log({
                "iteration": iteration,
                "train/success_rate": success_rate,
                "train/avg_questions": avg_questions,
                "train/avg_reward": avg_reward,
                "train/efficiency": avg_efficiency,
                **{f"ppo/{k}": v for k, v in stats.items() if isinstance(v, (int, float))}
            })
    
    def _log_eval_stats(self, iteration: int, results: dict):
        """Log evaluation statistics."""
        print(f"\n{'='*60}")
        print(f"EVALUATION at iteration {iteration}")
        print(f"Success Rate: {results['success_rate']:.2%}")
        print(f"Avg Questions: {results['avg_questions']:.1f}")
        print(f"Avg Reward: {results['avg_reward']:.2%}")
        print(f"{'='*60}\n")
  
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        save_path = f"{self.config.rl_output_dir}/{name}"
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Checkpoint saved to {save_path}")


# Main training script
if __name__ == "__main__":
    from config import TrainingConfig
    from data_utils import CityDataset

    CITIES = [
    'Guayaquil, Ecuador', 'Taipei, China', 'Zibo, China', 'Jinan, China', 'Alexandria, Egypt',
    'Berlin, Germany', 'Sydney, Australia', 'Istanbul, Turkey', 'Osaka, Japan', 'Hong Kong, China',
    'Bogota, Colombia', 'Jakarta, Indonesia', 'Bogor, Indonesia', 'Bandung, Indonesia', 'Kolkata, India',
    'Tashkent, Uzbekistan', 'Chengdu, China', 'Giza, Egypt', 'Semarang, Indonesia', 'Lima, Peru',
    'Hyderabad, India', 'Havana, Cuba', 'Harbin, China', 'Izmir, Turkey', 'Brasilia, Brazil',
    'Shenyang, China', 'Delhi, India', 'Baghdad, Iraq', 'Rio de Janeiro, Brazil', 'London, UK',
    'Rome, Italy', 'Los Angeles, USA', 'Mexico City, Mexico', 'Bucharest, Romania', 'Ho Chi Minh City, Vietnam',
    'Daegu, South Korea', 'Toronto, Canada', 'Surabaya, Indonesia', 'Bangalore, India', 'Fortaleza, Brazil',
    'Yokohama, Japan', 'Salvador, Brazil', 'St. Petersburg, Russia', 'Beijing, China', 'Wuhan, China',
    'Karachi, Pakistan', 'Cirebon, Indonesia', 'Dhaka, Bangladesh', 'Chicago, USA', 'Mumbai, India',
    'Guangzhou, China', 'Santiago, Chile', 'Budapest, Hungary', 'Tehran, Iran', 'Houston, USA',
    'Casablanca, Morocco', 'Kinshasa, Congo', 'Malang, Indonesia', 'Qingdao, China', "Xi'an, China",
    'Caracas, Venezuela', 'Abidjan, Côte d\'Ivoire', 'Medellin, Colombia', 'Tokyo, Japan', 'Chennai, India',
    'Kanpur, India', 'Bangkok, Thailand', 'Addis Ababa, Ethiopia', 'Busan, South Korea', 'Dalian, China',
    'Tianjin, China', 'Mashhad, Iran', 'Yangon, Myanmar', 'Sukabumi, Indonesia', 'Moscow, Russia',
    'Incheon, South Korea', 'Buenos Aires, Argentina', 'Cali, Colombia', 'New York, USA', 'Lahore, Pakistan',
    'Ahmedabad, India', 'Chongqing, China', 'Changchun, China', 'Nanjing, China', 'Madrid, Spain',
    'Taiyuan, China', 'Shanghai, China', 'Cairo, Egypt', 'Medan, Indonesia', 'Belo Horizonte, Brazil',
    'Paris, France', 'Nagoya, Japan', 'São Paulo, Brazil', 'Singapore, Singapore', 'Kiev, Ukraine',
    'Pyongyang, North Korea', 'Faisalabad, Pakistan', 'Ankara, Turkey', 'Quezon City, Philippines'
]

    
    # Load configuration
    config = TrainingConfig()
    
    if config.log_with == "wandb":

        wandb.init(
            project=config.project,
            name=config.name,
            config=config.__dict__
        )

    # Load city data
    city_dataset = CityDataset(CITIES)
    train_cities, test_cities = city_dataset.get_train_test_split(
        cities, 
        test_size=0.2
    )
    
    print(f"Loaded {len(train_cities)} training cities")
    print(f"Loaded {len(test_cities)} test cities")
    
    # Initialize trainer
    trainer = CityGuesserRLTrainer(
        base_model_path=config.base_model_path
        sft_model_path=config.sft_model_path,  # Your SFT model path
        cities_data=train_cities,
        config=config
    )
    
    # Train
    trainer.train(
        num_iterations=1000,
        episodes_per_iteration=8
    )