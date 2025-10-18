import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import numpy as np
from QADataset import QADataset
from environment import *
import tqdm
import wandb
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import random

cities = [
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
    'Paris, France', 'Nagoya, Japan', 'Sao Paulo, Brazil', 'Singapore, Singapore', 'Kiev, Ukraine',
    'Pyongyang, North Korea', 'Faisalabad, Pakistan', 'Ankara, Turkey', 'Quezon City, Philippines'
]


def generate_oracle_prompt(city, question):
    prompt = f""" Answer the following question about {city}.
    Do not reveal the name of {city} in your answer.
    Do not generate more than 50 words.
    Q:{question}
    A:"""
    return prompt

def transform_dataset(max_questions=20, batch_size=8, num_conversations=3000):
    

    with open('train.json', 'r') as file:
        data = json.load(file)
        pairs = []

        # Step 1: collect all histories for batch summarization
        all_histories = []
        history_mapping = []  # track (sample_idx, turn_idx)
        count = 0
        for sample_idx, sample in enumerate(data):
            if sample['correct']:
                count += 1
                if count > num_conversations:
                    break
                previous_qa = []
                for index, qa in enumerate(sample["lines"]):
                    questions_asked = index + 1
                    if questions_asked < max_questions and questions_asked < len(sample["lines"]):
                        q, a = qa.split("?", 1)
                        previous_qa.append(["Q: " + q.strip() + "?", "A: " + a.strip()])
                        all_histories.append(previous_qa.copy())
                        history_mapping.append((sample_idx, index))

        # Step 2: batch summarize all histories
        #all_summaries = batch_summarize(all_histories, model, tokenizer, batch_size=batch_size)
        all_contexts = []
        for history in all_histories:
            context = ""
            for qa in history:
                context += f"{qa[0]} {qa[1]}\n"
            all_contexts.append(context)    
 
        # Step 3: generate prompts using summaries
        context_idx = 0
        
        for (sample_idx, index) in history_mapping:
            next_question = f"{data[sample_idx]['lines'][index + 1].split('?')[0]}?"
            context = all_contexts[context_idx]
            context_idx += 1
            if index + 1 == max_questions - 1:
                pairs.append([context + "Make a final guess about the city.", next_question])
            else:
                pairs.append([context + "Ask the next question about the city. When you are confident you know its name, make a final guess.", next_question])
            
        # Step 4: add first question
        count = 0
        for sample in data:
            if sample['correct']:
                count += 1
                if count >  num_conversations:
                    break
                first_q = f"{sample['lines'][0].split('?')[0]}?"
                pairs.insert(0, [initial_prompt, first_q])
        return pairs

def train_model(pairs):
    #openai-community/gpt2-xl

    local_path = "./gpt2-medium-offline"   
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(local_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id


    lora_config = LoraConfig(
        r=8,               
        lora_alpha=32,  
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)

    run = wandb.init(
        project="research-multi-turn-RL-with-LMs",
        name="gpt2-medium-lora",
        config={
            "learning_rate": 5e-5,       # smaller LR for stability
            "batch_size": 4,
            "max_length": tokenizer.model_max_length,
            "epochs": 2,
            "optimizer": "AdamW",
            "accum_steps": 8,            # gradient accumulation
        },
    )

    dataset = QADataset(pairs, tokenizer, max_length=run.config["max_length"])  
    dataloader = DataLoader(dataset, batch_size=run.config["batch_size"], shuffle=True)

    optimizer = AdamW(model.parameters(), lr=run.config["learning_rate"])
    total_steps = len(dataloader) * run.config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * total_steps, num_training_steps=total_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    wandb.watch(model, log="all", log_freq=50)

    model.train()

    for epoch in tqdm.tqdm(range(run.config["epochs"]), desc="Epochs"):
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss / run.config["accum_steps"]  # scale loss
            loss.backward()

            # Gradient accumulation
            if (step + 1) % run.config["accum_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if (step + 1) % 100 == 0:
                wandb.log({
                    "loss": loss.item() * run.config["accum_steps"], 
                    "epoch": epoch,
                    "step": step + epoch * len(dataloader)
                })

    run.finish()

    model.save_pretrained("bc/fine_tuned_gpt2_medium_lora-2")
    tokenizer.save_pretrained("bc/fine_tuned_gpt2_medium_lora-2")

    """
    local_path = "./gpt2-medium-offline"   
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(local_path)

    run = wandb.init(
        project="research-multi-turn-RL-with-LMs",
        name="gpt2-medium-finetune",
        config={
            "learning_rate": 5e-5,
            "batch_size": 4,
            "max_length": tokenizer.model_max_length,
            "epochs": 1,
            "model_name": "gpt2-medium",
            "optimizer": "AdamW",
            "accum_steps": 8
        },
    )


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    dataset = QADataset(pairs, tokenizer, max_length=run.config["max_length"])  
    
    
    dataloader = DataLoader(dataset, batch_size=run.config["batch_size"], shuffle=True)

    optimizer = AdamW(model.parameters(), lr=run.config["learning_rate"])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    wandb.watch(model, log="all", log_freq=50)
    #scaler = GradScaler() 
    model.train()

    for epoch in tqdm.tqdm(range(run.config["epochs"]), desc="Epochs"):

        #model.train()
        #optimizer.zero_grad()
       
        for step, batch in enumerate(dataloader):
           # with autocast():

            outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device),
                )

           # is_last_batch = step + 1 == len(dataloader)
           # in_last_group = (step  // run.config["accum_steps"])  == (len(dataloader) // run.config["accum_steps"])
           # remaining = len(dataloader) % run.config["accum_steps"]
           # current_accum =  run.config["accum_steps"] if not (in_last_group and remaining != 0) else remaining

                # Scale loss accordingly
            loss = outputs.loss
            loss.backward()

            # Gradient accumulation
          #  if (step + 1) % run.config["accum_steps"] == 0 or is_last_batch:
                #scaler.step(optimizer)
                #scaler.update()
                #optimizer.zero_grad()
            optimizer.step()
            optimizer.zero_grad()

            #run.log({
            #    "loss": loss.item(), 
            #    "epoch": epoch,
            #    "step": step + epoch * len(dataloader)
            #})

            # Save checkpoint every 100 steps
            if (step + 1) % 100 == 0:
                ckpt_path = f"./checkpoints/epoch{epoch}_step{step+1}"  
                run.log({
                "loss": loss.item(), 
                "epoch": epoch,
                "step": step + epoch * len(dataloader)
                })
    

    run.finish()

    # Save final model
    model.save_pretrained("./fine_tuned_gpt2_medium")
    tokenizer.save_pretrained("./fine_tuned_gpt2_medium")
    """

def is_city_guess(question: str) -> bool:
    q = question.strip()
    prefix = "Is the city you are from"
    if not q.lower().startswith(prefix.lower()):
        return False

    # Find the character immediately after the prefix
    rest = q[len(prefix):].lstrip()
    if not rest:
        return False  # no text after prefix

    first_char = rest[0]
    # City guesses usually start with a capital letter, not lowercase
    return first_char.isupper()


@torch.no_grad()
def generate_trajectories(model, tokenizer, num_runs):
    trained_model = model.eval()
    trained_tokenizer = tokenizer
    oracle_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct").eval()
    oracle_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model.to(device)
    oracle_model.to(device)

    trajectories = []
    max_steps = 20

    trajectories = []
    random.seed(42)
    random_cities = random.sample(cities, 30)

    for _ in range(num_runs):
        for city in random_cities:
            target_city = city
            previous_qa = [] 
            context_str = ""
            questions = 0
            print(f"{target_city}\n")
            while questions < max_steps: #trained_model did not generate final guess:
                prompt = context_str + "Q:"
                context = trained_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024 - 30).to(device)
                input_ids = context['input_ids']
                question_ids = trained_model.generate(**context, do_sample=True, top_p=0.9, temperature=0.7, max_new_tokens=30, pad_token_id=trained_tokenizer.eos_token_id)
                question_ids = question_ids[:, input_ids.shape[1]:]  # Remove prompt tokens
                question = trained_tokenizer.decode(question_ids[0], skip_special_tokens=True)
                question = question.split("?")[0] + "?"
                question = question.strip()
                print(f"Q{questions + 1}: {question}")
                context = oracle_tokenizer(generate_oracle_prompt(target_city, question), return_tensors="pt").to(device)
                answer_ids = oracle_model.generate(**context, do_sample=False, max_new_tokens=100, pad_token_id=oracle_tokenizer.eos_token_id,  eos_token_id=oracle_tokenizer.convert_tokens_to_ids("Question"))
                input_ids = context['input_ids']
                answer_ids = answer_ids[:, input_ids.shape[1]:]  
                answer = oracle_tokenizer.decode(answer_ids[0], skip_special_tokens=True).strip()
                stop_cues = ["Question", "Answer"]
                # truncate at the first cue that appears
                for cue in stop_cues:
                    stop_idx = answer.find(cue)
                    if stop_idx != -1:
                        answer = answer[:stop_idx].strip()
                        break
                print(f"A{questions + 1}: {answer}")
                previous_qa.append([question, answer])
                context_str += f"Q:{question} A:{answer}\n"
                questions += 1
                if is_city_guess(question) or target_city.split(",")[0].lower() in question.lower():
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

def run_evaluation(model, tokenizer, step):
    trajectories = generate_trajectories(model, tokenizer, 1)
    rewards = reward_trajectories(trajectories)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    accuracy = np.sum([t['target_city'].lower() in t['questions'][-1].lower() for t in trajectories]) / len(trajectories)
    print(f"Average reward: {avg_reward}")
    print(f"Std reward: {std_reward}")
    print(f"Accuracy: {accuracy}")

    with open(f"bc/summary_bc_filtered2_{step}.out", "w") as f:
        f.write(f"Average reward: {avg_reward}\n")
        f.write(f"Std reward: {std_reward}\n")
        f.write(f"Accuracy: {accuracy}\n")

    with open(f"bc/trajectories_bc_filtered2_{step}.out", "w") as f:
        json.dump(trajectories, f, indent=2)
    

def train_model2(conversations):
    #openai-community/gpt2-xl

    local_path = "./gpt2-medium-offline"   
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForCausalLM.from_pretrained(local_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    lora_config = LoraConfig(
        r=8,               
        lora_alpha=32,  
        target_modules=["c_attn"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)

    run = wandb.init(
        project="research-multi-turn-RL-with-LMs",
        name="gpt2-medium-lora-bc-filtered",
        config={
            "learning_rate": 1e-4,       
            "batch_size": 4,
            "max_length": tokenizer.model_max_length,
            "epochs": 1,
            "optimizer": "AdamW",
            "accum_steps": 8,        
        },
    )

    dataset = QADataset(conversations, tokenizer, True)  
    dataloader = DataLoader(dataset, batch_size=run.config["batch_size"], shuffle=True)

    optimizer = AdamW(model.parameters(), lr=run.config["learning_rate"])
    total_steps = len(dataloader) * run.config["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * total_steps, num_training_steps=total_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    wandb.watch(model, log="all", log_freq=50)

    model.train()

    for epoch in tqdm.tqdm(range(run.config["epochs"]), desc="Epochs"):
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):

            input_ids = batch["input_ids"].to(device)
            should_take_action = batch["should_take_action"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids[:, 1:].clone()
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            should_take_action = should_take_action[:, 1:]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            # Compute per-token CE loss without reduction
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1).long(),
                reduction="none"
            ).view(labels.shape)  # [batch, seq_len]

            mask = should_take_action.float() * attention_mask
            n = mask.sum()  
            loss = (ce_loss * mask).sum() / n

            loss /= run.config["accum_steps"]  # scale loss
            loss.backward()

            # Gradient accumulation
            if (step + 1) % run.config["accum_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if (step + 1) % 100 == 0:
                wandb.log({
                    "loss": loss.item() * run.config["accum_steps"], 
                    "epoch": epoch,
                    "step": step + epoch * len(dataloader)
                })

            if (step + 1) % (len(dataloader) // 4) == 0:
                run_evaluation(model, tokenizer, step+1)

    run.finish()

    model.save_pretrained("bc/fine_tuned_gpt2_medium_lora_filtered2")
    tokenizer.save_pretrained("bc/fine_tuned_gpt2_medium_lora_filtered2")

    
if __name__ == "__main__":

    with open('train2.json', 'r') as f:
        conversations = json.load(f)

    train_model2(conversations)
    
    """
    pairs = transform_dataset()
    with open('new_training.json', 'w') as f:
         json.dump(pairs, f, indent=2)
    train_model(pairs)
    """
