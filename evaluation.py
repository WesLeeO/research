
import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch
import numpy as np
from peft import PeftModel, PeftConfig


cities = ['guayaquil,ecuador', 'taipei,china', 'zibo,china', 'jinan,china', 'alexandria,egypt', 
'berlin,germany', 'sydney,australia', 'istanbul,turkey', 'osaka,japan', 'hong kong,china', 'bogota,colombia', 'jakarta,indonesia', 
'bogor,indonesia', 'bandung,indonesia', 'calcutta,india', 'tashkent,uzbekistan', 
'chengdu,china', 'giza,egypt', 'semarang,indonesia', 'lima,peru', 'hyderabad,india', 
'havanna,cuba', 'haerbin,china', 'izmir,turkey', 'brasilia,brazil', 'shenyang,china', 'delhi,india', 
'baghdad,iraq', 'rio de janeiro,brazil', 'london,uk', 'rome,italy', 'los angeles,usa', 
'mexico city,mexico', 'bucuresti,romania', 'ho chi minh city,vietnam', 'taega,south korea', 
'toronto,canada', 'surabaya,indonesia', 'bangalore,india', 'fortaleza,brazil', 'yokohama,japan', 
'salvador,brazil', 'st petersburg,russia', 'beijing,china', 'wuhan,china', 'karachi,pakistan', 
'cirebon,indonesia', 'dhaka,bangladesh', 'chicago,usa', 'bombay,india', 'guangzhou,china', 
'santiago,chile', 'budapest,hungary', 'tehran,iran', 'houston,usa', 'casablanca,morocco', 'kinshaha,congo', 
'malang,indonesia', 'qingdao,china', 'xian,china', 'caracas,venezuela', 'abidjan,cote d’ivorie', 
'medellin,colombia', 'tokyo,japan', 'madras,india', 'kanpur,india', 'bangkok,thailand', 'addis ababa,ethopia', 
'pusan,south korea', 'dalian,china', 'tianjin,china', 'mashhad,iran', 'yangon,myanmar', 
'sukabumi,indonesia', 'moscow,russia', 'inchon,south korea', 'buenos aires,argentina', 'cali,colombia', 'new york,usa', 
'lahore,pakistan', 'ahmedabad,india', 'chongqing,china', 'changchun,china', 'nanjing,china', 'madrid,spain', 'taiyuan,china', 
'shanghai,china', 'cairo,egypt', 'medan,indonesia', 'belo horizonte,brazil', 'paris,france', 'nagoya,japan', 
'sao paulo,brazil', 'singapore,singapore', 'kiev,ukraine', 'pyong yang,north korea', 'faisalabad,pakistan', 'ankara,turkey', 'quezon city,philippines']


def generate_oracle_prompt(city, question):
    prompt = f""" Answer the following question about {city}.
    Do not reveal the name of {city} in your answer.
    Do not generate more than 50 words.
    Q:{question}
    A:"""
    return prompt

 
initial_prompt = (
    "You can ask up to 20 questions (the less the better) to determine the city chosen by an oracle."
    "Be smart and try to narrow down the possible cities as much as possible."
    "Generate the first question you will ask to determine the city. "
)


"""
def generate_intermediate_prompt(conversation_length, previous_qa):

    context = "\n".join([f"{qa[0]} {qa[1]}" for qa in previous_qa])
    prompt = (
        "You are playing a game of twenty questions. You can ask up to 20 questions (the less the better) "
        "to determine the city chosen by an oracle. Each turn, you can ask an open-ended question. "
        f"You have already asked {conversation_length} questions. "
        "You are smart, so you will ask the question that will narrow down the possible cities as "
        "much as possible. Don’t get stuck on one idea and try to branch out if you get stuck. "
        "Here are the questions you’ve asked and their corresponding answers: \n"
        f"{context}\n"
        "Based on what you know about the city so far, generate the next question you will ask to determine the city."
    )
    return prompt
"""

"""
def generate_final_prompt(previous_qa):
    context = "\n".join([f"{qa[0]} {qa[1]}" for qa in previous_qa])
    prompt = (
        "You are playing a game of twenty questions. You can ask up to 20 questions (the less the better) "
        "to determine the city chosen by an oracle. Each turn, you can ask an open-ended question. "
        f"You have already asked 19 questions, so this is your final guess. "
        "Here are the questions you’ve asked and their corresponding answers: \n"
        f"{context}\n"
        "Based on what you know about the city so far, generate the next question you will ask to determine the city. "
        "Only guess one city. Is the city "
    )
    return prompt
"""

def generate_intermediate_prompt(conversation_length, summary):
    answers = "\n".join([qa[1] for qa in summary])
    prompt = (
        f"Ask the next best question to narrow down the possible city you should guess. "
        "Do not ask a question if you already know its answer. "
        f"You have asked {conversation_length}/20 questions.\n"
        f"Facts known so far: {answers}\n"
        "Next question:"
    )
    return prompt

def generate_final_prompt(summary):
    answers = "\n".join([qa[1] for qa in summary])
    prompt = (
        "Make your final guess of the secret city. "
        "You have already asked 19/20 questions.\n"
        f"Facts known so far: {answers}\n"
        "Complete the following question: Is the city"
    )
    return prompt

def generate_trajectories(trained_model, trained_tokenizer, oracle_model, oracle_tokenizer,
                          cities, num_trajectories=1000, max_steps=20):
       
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model.to(device)
    oracle_model.to(device)
     # Test with a very basic completion
   # test_prompt = "The weather is"
   # test_context = trained_tokenizer(test_prompt, return_tensors="pt").to(device)
   # test_output = trained_model.generate(
   #     **test_context, 
   #     max_new_tokens=10,
   #     pad_token_id=trained_tokenizer.eos_token_id
   # )
   # print("Test completion:", trained_tokenizer.decode(test_output[0]))

    #load trained model
    trajectories = []
    for _ in range(num_trajectories):
        target_city = random.choice(cities)
        previous_qa = [] 
        context_str = ""
        questions = 0
        while questions < max_steps: #trained_model did not generate final guess:
            prompt = context_str + "Q:"
            """
            if questions == 0:
                prompt = initial_prompt
            else:
                if questions == max_steps - 1:
                    prompt = context_str + "Make a final guess about the city."
                else: 
                    prompt = context_str + "Ask the next question about the city. When you are confident you know its name, make a final guess."
            """
            context = trained_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024 - 30).to(device)
            input_ids = context['input_ids']
            question_ids = trained_model.generate(**context, do_sample=True, top_p=0.9, temperature=0.7, max_new_tokens=30, pad_token_id=trained_tokenizer.eos_token_id)
            question_ids = question_ids[:, input_ids.shape[1]:]  # Remove prompt tokens
            question = trained_tokenizer.decode(question_ids[0], skip_special_tokens=True)
            question = question.split("?")[0] + "?"
            question = question.strip()
            print(f"Q{questions + 1}: {question}")
            context = oracle_tokenizer(generate_oracle_prompt(target_city, question), return_tensors="pt").to(device)
            #answer_ids = oracle_model.generate(**context, do_sample=True, top_p=0.9, temperature=0.7, repetition_penalty=1.2, min_length=20, max_new_tokens=100, pad_token_id=oracle_tokenizer.eos_token_id)
            answer_ids = oracle_model.generate(**context, do_sample=False, max_new_tokens=50, pad_token_id=oracle_tokenizer.eos_token_id,  eos_token_id=oracle_tokenizer.convert_tokens_to_ids("Question"))
            input_ids = context['input_ids']
            answer_ids = answer_ids[:, input_ids.shape[1]:]  
            answer = oracle_tokenizer.decode(answer_ids[0], skip_special_tokens=True).strip()
            stop_idx = answer.find("Question")
            if stop_idx != -1:
                answer = answer[:stop_idx].strip()
            print(f"A{questions + 1}: {answer}")
            previous_qa.append([question, answer])
            context_str += f"Q:{question} A:{answer}\n"
            questions += 1
            if question.startswith("Is the city you are from") or target_city.split(",")[0].lower() in question.lower():
                break
        trajectories.append({"questions": [l[0] for l in previous_qa], "target_city": target_city.split(",")[0]})
    return trajectories

def reward_trajectories(trajectories):
    rewards = []
    for trajectory in trajectories:
        reward = -len(trajectory["questions"])
        final_question = trajectory["questions"][-1]
        if trajectory["target_city"].lower() in final_question.lower():
            reward += 1
        rewards.append(reward)
    return rewards


def baseline_reward():
    rewards = []
    with open('train.json', 'r') as file:
        data = json.load(file)
        for sample in data:
            reward = -len(sample["lines"])
            if sample['correct'] and sample['word'].split(',')[0].lower() in sample['lines'][-1].split('?')[0].lower():
                reward += 1
            rewards.append(reward)
    return np.mean(rewards)

def max_raw_return():
    max_reward = -100
    with open('train.json', 'r') as file:
        data = json.load(file)
        for sample in data:
            reward = -len(sample["lines"])
            if sample['word'].split(',')[0].lower() in sample['lines'][-1].split('?')[0].lower():
                reward += 1
            if reward > max_reward:
                max_reward = reward
    print(max_reward)
    return max_reward    

def min_raw_return():
    min_reward = 0
    with open('train.json', 'r') as file:
        data = json.load(file)
        for sample in data:
            reward = -len(sample["lines"])
            if sample['word'].split(',')[0].lower() in sample['lines'][-1].split('?')[0].lower():
                reward += 1
            if reward < min_reward:
                min_reward = reward
    print(min_reward)
    return min_reward    


def score(raw_return, dataset_average, min_raw_return=-20, max_raw_return=0):
    if raw_return >= dataset_average:
        return 50 + (raw_return - dataset_average) / (max_raw_return - dataset_average) * 50
    else:
        return (raw_return - min_raw_return) / (dataset_average - min_raw_return) * 50


def score_from_trajectories(path):
    with open(path, 'r') as f:
        trajectories = json.load(f)
        rewards = reward_trajectories(trajectories)
        return score(np.mean(rewards), baseline_reward(), min_raw_return=min_raw_return(), max_raw_return = max_raw_return())


if __name__ == "__main__":
    
    adapter_path = "/cluster/project/infk/krause/wnanadavies/GuessMyCity/bc/fine_tuned_gpt2_medium_lora" 
    base_model_path = "/cluster/project/infk/krause/wnanadavies/GuessMyCity/gpt2-medium-offline" 
    oracle_path = "/cluster/project/infk/krause/wnanadavies/GuessMyCity/oracle" 

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    oracle_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    oracle_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    

    trajectories = generate_trajectories(model, tokenizer, oracle_model, oracle_tokenizer, cities, num_trajectories=1)

    with open('bc/generated_trajectories2.json', 'w') as f:
        json.dump(trajectories, f)

    """
    #oracle_model_name = "/cluster/scratch/wnanadavies/GuessMyCity/gpt2-medium-offline" 
    oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_path, local_files_only=True)
    oracle_model = AutoModelForSeq2SeqLM.from_pretrained(oracle_path, local_files_only=True)

    trajectories = generate_trajectories(model, tokenizer, oracle_model, oracle_tokenizer, cities, num_trajectories=1)

    with open('bc/generated_trajectories2.json', 'w') as f:
        json.dump(trajectories, f)

    
    with open('generated_trajectories.json', 'w') as f:
        json.dump(trajectories, f)
    
    
    score = score_from_trajectories('generated_trajectories.json')
    print(score)
    #rewards = reward_trajectories(trajectories)
    #print(f"Average reward: {np.mean(rewards)}")
    #print(f"Std reward: {np.std(rewards)}")
    #print(f"Accuracy: {np.sum([t['target_city'].lower() in t['questions'][-1].lower() for t in trajectories])} / {len(trajectories)}")
    #print(f"Score: {score(raw_return=np.mean(rewards), dataset_average=dataset_average)}")
    """
