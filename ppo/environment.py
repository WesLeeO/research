import random
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


class LLMOracle:
    """LLM-based oracle that answers questions about a target city."""
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        print(f"Loading oracle model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def answer_question(self, question: str, city: str) -> str:
    
        prompt = self._create_oracle_prompt(question, city)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        answer = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip()

        stop_cues = ["\n", "Question", "Answer"]

        for cue in stop_cues:
            stop_idx = answer.find(cue)
            if stop_idx != -1:
                answer = answer[:stop_idx].strip()
                break

        return answer
    
    def _create_oracle_prompt(self, question: str, city: str) -> str:
        prompt = f""" Answer the following question about {city}.
        Do not reveal the name of {city} in your answer.
        Do not generate more than 50 words.
        Q:{question}
        A:"""
        return prompt
    

class CityGuessingEnvironment:
    """
    Environment for city guessing game with SPARSE REWARDS.
    """
    
    def __init__(
        self, 
        cities: List[str], 
        oracle: LLMOracle,
        max_questions: int = 20,
        question_penalty: float = -1.0,
        repetition_penalty: float = -1.0,
        no_guess_penalty: float = -10.0,
        correct_reward: float = 0.0,
    ):
        self.cities = cities
        self.oracle = oracle
        self.max_questions = max_questions
        
        # Reward parameters
        self.question_penalty = question_penalty  # -1 per question
        self.repetition_penalty = repetition_penalty
        self.no_guess_penalty = no_guess_penalty
        self.correct_reward = correct_reward
        
    def reset(self):
        self.target_city = random.choice(self.cities)
        print(self.target_city)
        self.questions_asked = 0
        self.history = []
        self.game_over = False

    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:

        action = action.strip()
        
        if self._is_guess(action):
            return self._process_guess(action)
        else:
            return self._process_question(action)
    """
    def _is_guess(self, action: str) -> bool:
        guess_indicators = [
            "Is the city you are from", 
            "This could help narrow down the possibilities about the city being"
        ]
        
        return any(indicator.lower() in action.lower() for indicator in guess_indicators)
    
    """

    def _is_guess(self, action: str) -> bool:
        """Check if action is a final guess."""
        city_guess_pattern = r"^Is the city you are from\s*[A-Z].*$"
        second_indicator = "This could help narrow down the possibilities about the city being"
        return bool(re.match(city_guess_pattern, action)) or second_indicator.lower() in action.lower()

    
    def _process_question(self, question: str) -> Tuple[Dict, float, bool, Dict]:

        self.questions_asked += 1
        answer = self.oracle.answer_question(question, self.target_city)
        done = self.questions_asked == self.max_questions
        reward = self.question_penalty
   
        for step in self.history:
            if question == step['question']:
                reward += self.repetition_penalty
                break

        if done: 
            reward += self.no_guess_penalty

        self.history.append({
            "type": "question",
            "question": question,
            "answer": answer,
            "reward": reward,
            "done": done,
            "correct": False,
            "asked": self.questions_asked
        })
        return self.history[-1]
    
    def _process_guess(self, guess: str) -> Tuple[Dict, float, bool, Dict]:
    
        self.game_over = True
        self.questions_asked += 1
        answer = self.oracle.answer_question(guess, self.target_city)
        correct = self.target_city.split(',')[0].lower() in guess.lower()
        reward = self.correct_reward if correct else self.question_penalty
 
        self.history.append({
            "type": "guess",
            "question": guess,
            "answer": answer,
            "reward": reward,
            "done": True,
            "correct": correct,
            "asked": self.questions_asked
        })

        return self.history[-1]
    