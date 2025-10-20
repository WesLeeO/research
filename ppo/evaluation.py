import torch
from typing import Dict, List
from tqdm import tqdm


class Evaluator:
    """Evaluator for the city guessing game."""
    
    def __init__(self, env, tokenizer, generations_kwargs):
        """
        Args:
            env: CityGuessingEnvironment instance
            tokenizer: Tokenizer for the model
            max_context_length: Maximum context length for truncation
        """
        self.env = env
        self.tokenizer = tokenizer
        self.max_context_length = tokenizer.model_max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_kwargs = generation_kwargs
    
    def evaluate(
        self, 
        model, 
        num_episodes: int = 20,
        temperature: float = 0.7,
        max_new_tokens: int = 50,
    ) -> Dict[str, float]:
        """
        Evaluate the model on multiple episodes.
        
        Args:
            model: The model to evaluate (with value head)
            num_episodes: Number of episodes to run
            temperature: Sampling temperature
            max_new_tokens: Max tokens to generate per question
            
        Returns:
            Dictionary with evaluation metrics
        """
        model.eval()  # Set to eval mode
        
        results = {
            "successes": 0,
            "total_questions": 0,
            "total_rewards": 0,
            "episodes": []
        }
        
        print(f"\nEvaluating on {num_episodes} episodes...")
        
        for episode_idx in tqdm(range(num_episodes), desc="Evaluation"):
            episode_result = self._run_episode(
                model, 
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                verbose=verbose
            )
            
            results["episodes"].append(episode_result)
            results["successes"] += int(episode_result["correct"])
            results["total_questions"] += episode_result["questions_used"]
            results["total_rewards"] += episode_result["total_reward"]
            
        
        # Calculate aggregate metrics
        metrics = {
            "success_rate": results["successes"] / num_episodes,
            "avg_questions": results["total_questions"] / num_episodes,
            "avg_reward": results["total_rewards"] / num_episodes,
        }
        
        model.train()  
        return metrics
    
    def _run_episode(
        self, 
        model, 
        temperature: float,
        max_new_tokens: int,
        verbose: bool
    ) -> Dict:
        """Run a single evaluation episode."""
        state = self.env.reset()
        
        done = False
        total_reward = 0
        questions_asked = 0
        
        while not done:
            # Create prompt
            prompt = self._create_prompt(state)
            
            # Tokenize with truncation
            context_tensor = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_context_length,
                truncation_side="left"
            ).squeeze().to(self.device)
            
            # Generate question
            with torch.no_grad():
                # Use the pretrained_model attribute to avoid value head during generation
                question_tensor = model.pretrained_model.generate(
                    context_tensor.unsqueeze(0),
                    return_prompt=False,
                    **self.generation_kwargs
                )            

            next_question = self.tokenizer.decode(
                question_tensor,
                skip_special_tokens=True
            ).strip()
   
            next_question = (question.split("?")[0] + "?").strip()

            # Take action
            state, reward, done, info = self.env.step(question)            
            total_reward += reward
            questions_asked += 1
        
        # Compile episode result
        episode_result = {
            "correct": info.get("correct", False),
            "questions_used": questions_asked,
            "total_reward": total_reward,
            "target_city": info.get("target_city", "unknown"),
            "efficiency": info.get("efficiency", 0.0),
            "timeout": info.get("timeout", False),
        }
        
        return episode_result
    
    def _create_prompt(self, state: dict) -> str:
        """Create prompt from state (same as training)."""
        prompt = ""
        for entry in state["history"]:
            if entry["type"] == "question":
                prompt += f"Q: {entry['content']}\nA: {entry['answer']}\n"
        
        prompt += "Q: "
        return prompt
    