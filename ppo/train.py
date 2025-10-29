import os
import torch
from torch import nn
from tqdm import tqdm
from typing import List, Tuple


from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead
from peft import PeftModel, LoraConfig, get_peft_model

from environment import CityGuessingEnvironment, LLMOracle
from reward_shaper import RewardShaper
from config import TrainingConfig
from evaluation import Evaluator
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def compute_token_logprobs_from_logits(logits: torch.FloatTensor, token_ids: torch.LongTensor) -> torch.FloatTensor:
    """
    logits: [B, L, V]
    token_ids: [B, L]
    returns: token_logprobs [B, L]
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return token_logprobs  # [B, L]


def compute_gae_batch(rewards: torch.Tensor,
                      values: torch.Tensor,
                      mask: torch.Tensor,
                      gamma: float = 0.99,
                      lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and discounted returns for a batch of token-level trajectories.
    Args:
        rewards: [B, L] token-level rewards (zero for non-reward tokens)
        values:  [B, L] value predictions (V(s_t) for each token position)
        mask:    [B, L] binary mask (1 where token exists and valid; 0 for pad)
        gamma, lam: hyperparams
    Returns:
        advantages: [B, L]
        returns:    [B, L] (advantages + values)
    Notes:
        - mask should be 1 for real tokens (context or action); for GAE we *only* consider timesteps where mask==1.
        - typically you will set rewards only on action tokens (e.g., last token of a question) and zero elsewhere.
    """

    B, L = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Flatten valid tokens and compute per-batch offsets
    filtered_rewards = rewards[mask]   # [N_valid]
    filtered_values = values[mask]     # [N_valid]
    counts = mask.sum(dim=1).cpu().tolist()

    start = 0
    for b, length in enumerate(counts):
        if length == 0:
            continue
        end = start + length

        r = filtered_rewards[start:end]
        v = filtered_values[start:end]
        lastgaelam = 0
        adv = torch.zeros_like(r)

        # backward recursion over valid (agent) tokens
        for t in reversed(range(length)):
            next_value = v[t + 1] if t + 1 < length else 0.0
            delta = r[t] + gamma * next_value - v[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam

        # assign back to original tensor
        advantages[b][mask[b].bool()] = adv
        returns[b][mask[b].bool()] = adv + v

        start = end

    return advantages, returns

    """
    B, L = rewards.shape
 

    advantages = torch.zeros_like(rewards, dtype=torch.float32, device=rewards.device)
    returns = torch.zeros_like(rewards, dtype=torch.float32, device=rewards.device)

    # We need a "next value" for t = L-1, treat it as 0 when next mask == 0
    for b in range(B):
        lastgaelam = 0.0
        # iterate backwards over timesteps
        for t in reversed(range(L)):
            if mask[b, t] == 0:
                # padded step — reset
                lastgaelam = 0.0
                advantages[b, t] = 0.0
                returns[b, t] = 0.0
                continue
            next_mask = mask[b, t + 1] if t + 1 < L else 0
            next_value = values[b, t + 1] if (t + 1 < L and next_mask) else 0.0
            delta = rewards[b, t] + gamma * next_value * next_mask - values[b, t]
            lastgaelam = delta + gamma * lam * next_mask * lastgaelam
            advantages[b, t] = lastgaelam
            returns[b, t] = rewards[b, t] + gamma * returns[b, t + 1]
    return advantages, returns
    """

class CityGuesserRLTrainer:
    """RL trainer for GuessMyCity using TRL PPO."""

    def __init__(self, base_model_path: str, sft_model_path: str, cities: List[str], config: TrainingConfig):
        self.config = config
        self.cities = cities

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True)
        policy_model = PeftModel.from_pretrained(base_model, sft_model_path)
        self.policy_model = policy_model 
        policy_model.config.output_hidden_states = True
        self.value_head = nn.Linear(self.policy_model.config.hidden_size, 1)

        for name, p in self.policy_model.named_parameters():
            if "lora" in name:
                p.requires_grad = True


        """
        # Reference model (for KL)
        self.ref_model = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True),
            sft_model_path
        )
        """

        # Move to GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_model.to(self.device, dtype=torch.bfloat16)
        self.value_head.to(self.device, dtype=torch.bfloat16)
        #self.ref_model.to(self.device)

        # Environment & Oracle
        oracle = LLMOracle(model_name=config.oracle_model)
        self.env = CityGuessingEnvironment(
            cities=cities,
            oracle=oracle,
            max_questions=config.max_questions
        )

        """
        # Reward shaper
        self.reward_shaper = RewardShaper(config)
        self.reward_model = RewardModel(self.reward_shaper, self.tokenizer)

        # PPO Config
        ppo_config = PPOConfig(
            learning_rate=config.rl_learning_rate,
            batch_size=config.rl_batch_size,
            mini_batch_size=config.rl_mini_batch_size,
            num_ppo_epochs=config.num_ppo_epochs,
            max_grad_norm=config.max_grad_norm,
            kl_coef=config.kl_coef,
            gamma=config.gamma,
            lam=config.lam,
        )

        dummy_data = {"input_ids": [self.tokenizer.encode("Q: Start", return_tensors="pt").squeeze().tolist()]}
        self.dummy_dataset = Dataset.from_dict(dummy_data)

        # PPO Trainer
        self.ppo_trainer = PPOTrainer(
            args=ppo_config,
            processing_class=self.tokenizer,
            model=self.policy_model,
            train_dataset = self.dummy_dataset,
            ref_model=self.ref_model,
            value_model=self.value_model,
            reward_model=self.reward_model
        )
        """
        # Generation kwargs
        self.generation_kwargs = {
            "max_new_tokens": config.max_new_tokens_question,
            "temperature": config.temperature,
            "do_sample": True,
            "top_p": config.top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Evaluator
        self.evaluator = Evaluator(self.env, self.tokenizer, self.generation_kwargs)

    def create_prompt(self, env) -> str:
        prompt = ""
        for step in env.history:
            prompt += f"Q:{step['question']} A:{step['answer']}\n"
        prompt += "Q:"
        return prompt

    def play_episode(self):
        """Play one episode and collect rollout data for PPO."""
        self.env.reset()
        print('-------------------------------')
        token_ids = []
        attention_mask = []
        action_mask = []
        token_rewards = []

        self.tokenizer.truncation_side = "left"

        done = False
        while not done:
            prompt = self.create_prompt(self.env)
            print(f'Prompt: {prompt} \n')
            # We only use context to generate; we DO NOT append raw context tokens to trajectory.
            # Instead we append the generated question tokens + environment answer tokens.
            trained_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024 - 30).to(device)
            context_tensor = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_length - 30
            ).squeeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.policy_model.generate(
                    input_ids=context_tensor.unsqueeze(0),
                    **self.generation_kwargs
                )
                generated = outputs[:, context_tensor.size(0):] 

            next_question = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            next_question = (next_question.split("?")[0] + "?").strip()

            info = self.env.step(next_question)
            done = info['done']
            answer_text = info['answer']
            reward = info['reward']

            print(next_question)
            print(answer_text)

            q_tokens = self.tokenizer.encode(f"Q:{next_question}", add_special_tokens=False)
            a_tokens = self.tokenizer.encode(f"A:{answer_text}", add_special_tokens=False)

            token_ids.extend(q_tokens + a_tokens)
            attention_mask.extend([1] * (len(q_tokens) + len(a_tokens)))
            action_mask.extend([1] * len(q_tokens) + [0] * len(a_tokens))
            token_rewards.extend([0.0] * (len(q_tokens) - 1) + [reward] + [0.0] * len(a_tokens))

        # convert to tensors (1D)
        token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        print('generated_tokens shape ', token_ids.shape)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        action_mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
        token_rewards = torch.tensor(token_rewards, dtype=torch.float32, device=self.device)

        return token_ids, attention_mask, action_mask, token_rewards, info  # return final info for logging


    def collect_batch(self, num_episodes: int):

        token_id_list = []
        attention_mask_list = [] 
        action_mask_list = []
        reward_list = []
        infos = []

        for _ in range(num_episodes):
            token_ids, attention_mask, action_mask, rewards, info = self.play_episode()
            token_id_list.append(token_ids)
            attention_mask_list.append(attention_mask)
            action_mask_list.append(action_mask)
            reward_list.append(rewards)
            infos.append(info)

        token_ids = pad_sequence(token_id_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        action_mask = pad_sequence(action_mask_list, batch_first=True, padding_value=0)
        rewards = pad_sequence(reward_list, batch_first=True, padding_value=0)

        return {
            "input_ids": token_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "action_mask": action_mask.to(self.device),
            "rewards": rewards.to(self.device),
            "infos": infos
        }

    def train(self, num_iterations: int, episodes_per_iteration: int = 4):

        policy_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        value_params = [p for p in self.value_head.parameters()]
        policy_optim = torch.optim.AdamW(policy_params, lr=self.config.rl_learning_rate)
        value_optim = torch.optim.AdamW(value_params, lr=self.config.value_learning_rate)

        for iteration in tqdm(range(num_iterations)):
            batch = self.collect_batch(episodes_per_iteration)
            input_ids = batch["input_ids"]               # [B, L]
            attention_mask = batch["attention_mask"]     # [B, L]
            action_mask = batch["action_mask"]           # [B, L] (1.0 on question tokens)
            rewards = batch["rewards"]                   # [B, L]
            infos = batch["infos"]

            max_ctx = self.policy_model.config.n_positions

            print('still large? ', input_ids.size(1)) # why so large

            if input_ids.size(1) > max_ctx:
                input_ids = input_ids[:, -max_ctx:]              
                attention_mask = attention_mask[:, -max_ctx:]       
                action_mask = action_mask[:, -max_ctx:]           
                rewards = rewards[:, -max_ctx:]        

            B, L = input_ids.shape

            # ---- 1) Evaluate old policy/value (no grad) ----
            with torch.no_grad():
                policy_out = self.policy_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                old_logits = policy_out.logits              # [B, L, V]
                old_logprobs = compute_token_logprobs_from_logits(old_logits, input_ids)  # [B, L]
                old_values = self.value_head(policy_out.hidden_states[-1]).squeeze(-1)
            
            gae_mask = attention_mask * action_mask
            advantages, returns = compute_gae_batch(rewards, old_values, gae_mask.bool(), self.config.gamma, self.config.lam)
            
            # Normalize advantages over action positions only
            adv_mask = (action_mask == 1) & (attention_mask == 1)
            flat_adv = advantages[adv_mask]
            if flat_adv.numel() > 0:
                adv_mean = flat_adv.mean()
                adv_std = flat_adv.std(unbiased=False) + 1e-8
                advantages = (advantages - adv_mean) / adv_std

            for epoch in range(self.config.num_ppo_epochs):        
                # ---- 3) Forward current policy/value to get new logprobs & values ----
                policy_out_new = self.policy_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                logits_new = policy_out_new.logits
                new_logprobs = compute_token_logprobs_from_logits(logits_new, input_ids)  # [B, L]
                new_values = self.value_head(policy_out_new.hidden_states[-1]).squeeze(-1)

                # ---- 4) Compute PPO losses only on action tokens ----
                # select action positions
                mask = action_mask * attention_mask  # float mask
                denom = mask.sum()

                # Policy loss (clipped)
                log_ratio = (new_logprobs - old_logprobs) * mask
                ratio = torch.exp(log_ratio)
                surr1 = ratio * advantages * mask
                surr2 = torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange) * advantages * mask
                policy_loss = -(torch.sum(torch.min(surr1, surr2)) / denom)

                # Value loss (clipped)
                value_clipped = old_values + torch.clamp(new_values - old_values, -self.config.value_clip_epsilon, self.config.value_clip_epsilon)
                vf_loss1 = (new_values - returns) ** 2
                vf_loss2 = (value_clipped - returns) ** 2
                value_loss = torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / denom

                total_loss = policy_loss + self.config.vf_coef * value_loss
               
                policy_optim.zero_grad()
                value_optim.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_params, max_norm=self.config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(value_params, max_norm=self.config.max_grad_norm)
                policy_optim.step()
                value_optim.step()


            # ---- 6) Logging & eval ----
            if iteration % getattr(self.config, "log_every", 1) == 0:
                success = sum(1 for inf in infos if inf.get("correct", False))
                avg_qs = sum(inf.get("asked", 0) for inf in infos) / len(infos)
                print(f"Iter {iteration} | Loss {total_loss.item():.4f} | Policy {policy_loss.item():.4f} | Value {value_loss.item():.4f} | Success {success}/{len(infos)} | AvgQs {avg_qs:.1f}")
                wandb.log({
                    "total_loss": total_loss.item(), 
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "sucess_rate": success/len(infos),
                    "step": step + iteration * episodes_per_iteration
                })

            """
            if iteration % getattr(self.config, "eval_every", 10) == 0:
                results = self.evaluator.evaluate(self.policy_model, num_episodes=20)
                self._log_eval_stats(iteration, results)
            """

            if iteration % getattr(self.config, "save_every", 50) == 0:
                self.save_checkpoint(f"ppo_model_iter_{iteration}")

    
        self.save_checkpoint("ppo_model")
        print("Training complete!")

    def _log_training_stats(self, iteration, infos, stats):
        success_rate = sum(info["correct"] for info in infos) / len(infos)
        avg_questions = sum(info["questions_used"] for info in infos) / len(infos)
        print(f"Iteration {iteration} | Success: {success_rate:.2%} | Avg Qs: {avg_questions:.1f}")
        if stats:
            print(f"PPO Policy Loss: {stats.get('ppo/loss/policy',0):.4f}, Value Loss: {stats.get('ppo/loss/value',0):.4f}")

    def _log_eval_stats(self, iteration, results):
        print(f"Evaluation Iter {iteration} | Success: {results['success_rate']:.2%}")

    def save_checkpoint(self, name: str):
        path = os.path.join(self.config.rl_output_dir, name)
        self.policy_model.save_pretrained(path)
        self.value_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Checkpoint saved to {path}")
    

if __name__ == "__main__":
    from config import TrainingConfig
    import wandb
    from dataclasses import asdict

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
    'Casablanca, Morocco', 'Kinshasa, Congo', 'Malang, Indonesia', 'Qingdao, China', 'Xi’an, China',
    'Caracas, Venezuela', 'Abidjan, Côte d’Ivoire', 'Medellin, Colombia', 'Tokyo, Japan', 'Chennai, India',
    'Kanpur, India', 'Bangkok, Thailand', 'Addis Ababa, Ethiopia', 'Busan, South Korea', 'Dalian, China',
    'Tianjin, China', 'Mashhad, Iran', 'Yangon, Myanmar', 'Sukabumi, Indonesia', 'Moscow, Russia',
    'Incheon, South Korea', 'Buenos Aires, Argentina', 'Cali, Colombia', 'New York, USA', 'Lahore, Pakistan',
    'Ahmedabad, India', 'Chongqing, China', 'Changchun, China', 'Nanjing, China', 'Madrid, Spain',
    'Taiyuan, China', 'Shanghai, China', 'Cairo, Egypt', 'Medan, Indonesia', 'Belo Horizonte, Brazil',
    'Paris, France', 'Nagoya, Japan', 'São Paulo, Brazil', 'Singapore, Singapore', 'Kiev, Ukraine',
    'Pyongyang, North Korea', 'Faisalabad, Pakistan', 'Ankara, Turkey', 'Quezon City, Philippines']

    config = TrainingConfig()
    config_dict = asdict(config)

    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict
    )

    trainer = CityGuesserRLTrainer(
        base_model_path=config.base_model_path,
        sft_model_path=config.sft_model_path,
        cities=CITIES,
        config=config
    )
    trainer.train(num_iterations=100, episodes_per_iteration=4)