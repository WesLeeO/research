import random
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMOracle:
    """LLM-based oracle that answers questions about a target city."""
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):

        print(f"Loading oracle model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
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

        stop_cues = ["Question", "Answer"]

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
        correct_reward: float = 0.0
    ):
        self.cities = cities
        self.oracle = oracle
        self.max_questions = max_questions
        
        # Reward parameters
        self.question_penalty = question_penalty  # -1 per question
        self.correct_reward = correct_reward
        self.reset()
        
    def reset(self) -> Dict:
        """Reset environment for new episode."""
        self.target_city = random.choice(self.cities)
        self.questions_asked = 0
        self.history = []
        self.game_over = False
        return self._get_state()
    
    def _get_state(self) -> Dict:
        """Return current state information."""
        return {
            "questions_remaining": self.max_questions - self.questions_asked,
            "history": self.history.copy(),
            "target_city_name": self.target_city, 
        }
    
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Process an action (question or guess).
        
        Returns:
            state, reward, done, info
        """
        action = action.strip()
        
        # Detect if this is a guess
        if self._is_guess(action):
            return self._process_guess(action)
        else:
            return self._process_question(action)
    
    def _is_guess(self, action: str) -> bool:
        """Check if action is a final guess."""
    
        guess_indicators = [
            "Is the city you are from", 
            "This could help narrow down the possibilities about the city being"
        ]
        
        return any(indicator.lower() in action.lower() for indicator in guess_indicators)
    
    def _process_question(self, question: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Process a yes/no question.
        
        Returns -1 reward for every question.
        """

        #if self.questions_asked >= self.max_questions:
        #   # Out of questions, force guess
        #    return self._force_guess()
        
        self.questions_asked += 1
        answer = self.oracle.answer_question(question, self.target_city)
        
        self.history.append({
            "type": "question",
            "content": question,
            "answer": answer
        })
        
        # SPARSE REWARD: -1 for every question
        reward = self.question_penalty
        state = self._get_state()
        done = self.questions_asked == self.max_questions
        info = {
            "answer": answer,
            "question_num": self.questions_asked,
            "question": question
        }
        
        return state, reward, done, info
    
    def _process_guess(self, guess: str) -> Tuple[Dict, float, bool, Dict]:
        """
        Process final guess.
        
        Returns:
        - Correct: positive reward + efficiency bonus
        - Wrong: negative penalty
        """
        self.game_over = True
        """
        # Extract city name from guess
        guessed_city = self._extract_city_name(guess)
        
        # Check if correct
        correct = self._check_correct_guess(guessed_city, self.target)
        """
        correct = self.target_city.split(',')[0].lower() in guess

        if correct:
            # Correct guess: base reward + efficiency bonus
            """
            efficiency = (self.max_questions - self.questions_asked) / self.max_questions
            reward = self.correct_reward + (self.efficiency_bonus * efficiency)
            """
            reward = self.correct_reward
        else:
            # Wrong guess: penalty
            reward = self.wrong_penalty
        
        self.history.append({
            "type": "guess",
            "content": guess,
            "correct": correct
        })
        
        state = self._get_state()
        info = {
            "correct": correct,
            "target_city": self.target_city,
           # "guessed_city": guessed_city,
            "questions_used": self.questions_asked,
            "efficiency": 1.0 - (self.questions_asked / self.max_questions),
            "final_reward": reward
        }
        
        return state, reward, True, info
    
    
  
    """
    def _force_guess(self) -> Tuple[Dict, float, bool, Dict]:
        random_city = random.choice(self.cities_data)
        guess = f"Is the city {random_city['name']}?"
        return self._process_guess(guess)
    """

# Test the environment
if __name__ == "__main__":
    # Sample cities
    cities = [
        {
            "name": "Paris", "country": "France", "continent": "Europe",
            "population": 2161000, "coastal": False, "is_capital": True
        },
        {
            "name": "Tokyo", "country": "Japan", "continent": "Asia",
            "population": 13960000, "coastal": True, "is_capital": True
        },
    ]
    
    # Initialize oracle (this will download model if not cached)
    oracle = LLMOracle(model_name="gpt2")  # Use smaller model for testing
    
    # Create environment
    env = CityGuessingEnvironment(
        cities_data=cities,
        oracle=oracle,
        max_questions=10
    )
    
    # Test episode
    state = env.reset()
    print(f"Target: {state['target_city_name']}\n")
    
    # Simulate some actions
    actions = [
        "Is it in Europe?",
        "Is the population over 5 million?",
        "Is the city Paris?"
    ]
    
    total_reward = 0
    for action in actions:
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Action: {action}")
        if "answer" in info:
            print(f"Answer: {info['answer']}")
        print(f"Reward: {reward}")
        print(f"Total Reward: {total_reward}\n")
        
        if done:
            print(f"Game Over! Correct: {info['correct']}")
            print(f"Final Reward: {info.get('final_reward')}")
            break