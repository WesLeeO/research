import torch
from typing import Dict, List
from tqdm import tqdm
import random


class Evaluator:
    """Evaluator for the city guessing game."""
    
    def __init__(self, env, tokenizer, generation_kwargs, cities, config):
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
        self.cities = cities
        self.config = config
    
    def evaluate(self, model, num_episodes) -> Dict[str, float]:
  
        model.eval()  # Set to eval mode
        
        results = {
            "correct": 0,
            "asked": 0,
            "trajectories": []
        }

        random.seed(42)
        random_cities = random.sample(self.cities, 30)
        
        print(f"\nEvaluating on {num_episodes} episodes...")
        
        for city in random_cities:
            episode_result, questions = self._run_episode(model, city)
            results["correct"] += int(episode_result["correct"])
            results["asked"] += episode_result["asked"]
            results["trajectories"].append(questions)
            
        metrics = {
            "accuracy": results["correct"] / num_episodes,
            "avg": results["asked"] / num_episodes,
            "trajectories": results["trajectories"]
        } 
        return metrics
    
    def _run_episode(self, model, city) -> Dict:

        model.to(self.device)
        
        done = False
        total_reward = 0
        questions_asked = 0

        self.env.reset()
        self.env.target_city = city

        self.tokenizer.truncation_side = "left"
        info = None

        questions = []
        done = False
        print(city)
        print('--------------------')
        while not done:
            prompt = self.create_prompt(self.env)
            context_tensor = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_context_length - 30
            ).squeeze(0).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=context_tensor.unsqueeze(0),
                    **self.generation_kwargs
                )
                generated = outputs[:, context_tensor.size(0):] 

            next_question = self.tokenizer.decode(generated[0], skip_special_tokens=True).strip()
            next_question = (next_question.split("?")[0] + "?").strip()
            questions.append(next_question)
            info = self.env.step(next_question)
            done = info['done']
            answer = info['answer']
            print(next_question)
            print(answer)
        print('--------------------')
        return info, questions
    
    def create_prompt(self, env) -> str:
        prompt = ""
        for step in env.history:
            prompt += f"Q:{step['question']} A:{step['answer']}\n"
        prompt += "Q:"
        return prompt
    