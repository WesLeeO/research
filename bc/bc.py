import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch
import numpy as np
from QADataset import QADataset
from QADataset2 import QADataset2
from environment import *
import tqdm
import wandb
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F

"""
initial_prompt = (
    "You can ask up to 20 questions (the less the better) to determine the city chosen by an oracle."
    "Be smart and try to narrow down the possible cities as much as possible."
    "Generate the first question you will ask to determine the city. "
)

def batch_summarize(histories, model, tokenizer, batch_size=8, max_new_tokens=600):
    summaries = []
    for i in range(0, len(histories), batch_size):
        batch = histories[i:i+batch_size]
        prompts = [
         (
          "Instructions: \nYou are summarizing answers from an oracle about a secret city. "
          "Extract and summarize ONLY the information provided in the data below. "
          "DO NOT add any external knowledge, inferences, or creative details. \n "
          "- Ignore irrelevant, repeated, or uninformative text\n"
          "- Focus on location, climate, landmarks, culture, cuisine, events, or unique characteristics\n"
          "- Do not invent or extrapolate any information not explicitly stated\n" 
          "Present the information as concisely as possible (not more than 400 words).\n\n"
          "Answers:\n" + "\n".join([a for _, a in h])          
         )
          for h in batch
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id) # return_dict_in_generate=True, output_scores=True)

        # Extract only the new tokens
        batch_summaries = [
          tokenizer.decode(out, skip_special_tokens=True) for out in outputs
          #for out in outputs.sequences[:, inputs.input_ids.shape[1]:]
        ]

        summaries.extend(batch_summaries)
    return summaries

def generate_intermediate_prompt(conversation_length, summary):
    prompt = (
        f"Ask the next best question to narrow down the possible city you should guess. "
        f"You have asked {conversation_length}/20 questions.\n"
        f"Facts known so far: {summary}\n"
        "Next question:"
    )
    return prompt

def generate_final_prompt(summary):
    prompt = (
        "Make your final guess of the secret city. "
        "You have already asked 19/20 questions.\n"
        f"Facts known so far: {summary}\n"
        "Complete the following question: Is the city"
    )
    return prompt

""" 

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
        name="gpt2-medium-lora",
        config={
            "learning_rate": 1e-4,       
            "batch_size": 4,
            "max_length": tokenizer.model_max_length,
            "epochs": 2,
            "optimizer": "AdamW",
            "accum_steps": 8,        
        },
    )

    dataset = QADataset2(conversations, tokenizer)  
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

    run.finish()

    model.save_pretrained("bc/fine_tuned_gpt2_medium_lora")
    tokenizer.save_pretrained("bc/fine_tuned_gpt2_medium_lora")

    
if __name__ == "__main__":

    with open('train.json', 'r') as f:
        conversations = json.load(f)

    train_model2(conversations)
    
    """
    pairs = transform_dataset()
    with open('new_training.json', 'w') as f:
         json.dump(pairs, f, indent=2)
    train_model(pairs)
    """
