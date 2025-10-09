import torch.nn.functional as F
from GPT2MC import GPT2MC
import torch
from torch import optim
from MCDataset import MCDataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
import json
from torch.nn.utils.rnn import pad_sequence
import wandb
import tqdm
from BiasLogitsProcessor import BiasLogitsProcessor
from peft import PeftModel
import random
import numpy as np


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

    
def get_query_indicators(flat_mask):
    
    true_idxs = torch.nonzero(flat_mask, as_tuple=False)[:, 0]
    # Pad to the same size as flat_mask, filling with flat_mask.shape[0]
    num_pad = flat_mask.shape[0] - true_idxs.shape[0]
    if num_pad > 0:
        padding = torch.full((num_pad,), flat_mask.shape[0], dtype=true_idxs.dtype, device=flat_mask.device)
        idxs = torch.cat([true_idxs, padding], dim=0)
    else:
        idxs = true_idxs
    # One-hot encoding
    query_indicators = torch.nn.functional.one_hot(idxs, num_classes=flat_mask.shape[0]+1).float()[:, :-1]
    return query_indicators

def mc_loss(q_values, q_logits, token_ids, attention_mask, should_take_action, returns, cql_weight=0.01):

    """
    q_values: [batch, seq_len, vocab_size]  (Q-head outputs)
    q_logits: [batch, seq_len, vocab_size]  (LM logits)
    token_ids: [batch, seq_len]             (actions taken)
    attention_mask: [batch, seq_len]
    should_take_action: [batch, seq_len]    (binary mask for question tokens)
    returns: [batch, seq_len]               (target rewards/returns)
    """

    # shift for next-token prediction
    q_values = q_values[:, :-1, :]        
    q_logits = q_logits[:, :-1, :]
    token_ids = token_ids[:, 1:]     
    attention_mask = attention_mask[:, 1:]
    should_take_action = should_take_action[:, 1:]
    returns = returns[:, 1:]

    # mask for positions we care about
    mask = (should_take_action * attention_mask).float()  # [batch, seq_len-1]
    n = mask.sum()

    # ===== Q(s,a) selection =====
    # gather Q-values corresponding to the action actually taken
    # q_values: [batch, seq_len, vocab_size]
    # token_ids: [batch, seq_len] (indices)
    q_sa = q_values.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len - 1]

    # RTG regression loss
    q_loss = ((q_sa - returns) ** 2 * mask).sum() / n

    # CQL / cross-entropy loss to penalize unseen actions
    ce_loss = F.cross_entropy(
        q_logits.reshape(-1, q_logits.shape[-1]), 
        token_ids.reshape(-1).long(), 
        reduction='none'
    ).reshape(token_ids.shape)  # [batch, seq_len]

    q_cql_loss = (ce_loss * mask).sum() / n

    total_loss = q_loss + cql_weight * q_cql_loss

    return total_loss, dict(q_loss=q_loss.item(), q_cql_loss=q_cql_loss.item())

@torch.no_grad()
def generate_trajectories(gpt2MC, num_trajectories):
    trained_model = gpt2MC.model.eval()
    trained_tokenizer = gpt2MC.tokenizer
    oracle_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct").eval()
    oracle_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model.to(device)
    gpt2MC.q_function.to(device)
    oracle_model.to(device)

    trajectories = []
    max_steps = 20

    for _ in range(num_trajectories):
        target_city = random.choice(cities)
        context_str = ""
        previous_qa = []
        questions = 0

        while questions < max_steps:
            prompt = context_str + "Q:"
            context = trained_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024 - 30).to(device)
            input_ids = context["input_ids"]
            initial_length = input_ids.size(1)

            # Efficient autoregressive decoding loop
            
            outputs = trained_model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
                output_hidden_states=True
            )

            past_key_values = outputs.past_key_values

            for _ in range(30):
                last_hidden = outputs.hidden_states[-1][:, -1, :]     # (batch, hidden)
                bias = gpt2MC.q_function(last_hidden)                 # (batch, vocab)
                scores = outputs.logits[:, -1, :]                     # (batch, vocab)
                probs = F.softmax(scores, dim=-1)
                alpha = -1.0 ##???
                probs = probs ** (alpha * bias)
                probs = F.softmax(scores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token.item() == trained_tokenizer.eos_token_id:
                    break

                outputs = trained_model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_key_values = outputs.past_key_values

            question_ids = input_ids[:, initial_length:]
            question = trained_tokenizer.decode(question_ids[0], skip_special_tokens=True)
            question = question.split("?")[0].strip() + "?"
            print(f"Q{questions + 1}: {question}")
            # --- Generate answer from oracle ---
            oracle_prompt = generate_oracle_prompt(target_city, question)
            oracle_context = oracle_tokenizer(oracle_prompt, return_tensors="pt").to(device)
            answer_ids = oracle_model.generate(
                **oracle_context,
                do_sample=False,
                max_new_tokens=50,
                pad_token_id=oracle_tokenizer.eos_token_id,
                eos_token_id=oracle_tokenizer.convert_tokens_to_ids("Question")
            )
            ans_ids = answer_ids[:, oracle_context["input_ids"].shape[1]:]
            answer = oracle_tokenizer.decode(ans_ids[0], skip_special_tokens=True).strip()
            stop_idx = answer.find("Question")
            if stop_idx != -1:
                answer = answer[:stop_idx].strip()
            print(f"A{questions + 1}: {answer}")
            previous_qa.append([question, answer])
            context_str += f"Q:{question} A:{answer}\n"
            questions += 1

            if question.startswith("Is the city you are from") or target_city.split(",")[0].lower() in question.lower():
                break

        trajectories.append({
            "questions": [q for q, _ in previous_qa],
            "target_city": target_city.split(",")[0]
        })

    return trajectories
    """
    trained_model = gpt2MC.model
    trained_tokenizer = gpt2MC.tokenizer
    oracle_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    oracle_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")  

    logits_processor = BiasLogitsProcessor(gpt2MC)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trained_model.to(device)
    gpt2MC.q_function.to(device)
    oracle_model.to(device)
    trajectories = []
    max_steps = 20
    
    for _ in range(num_trajectories):
        target_city = random.choice(cities)
        previous_qa = [] 
        context_str = ""
        questions = 0
        while questions < max_steps: 
            prompt = context_str + "Q:"
            context = trained_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024 - 30).to(device)
            input_ids = context['input_ids']
            question_ids = trained_model.generate(**context, do_sample=True, top_p=0.9, temperature=0.7, max_new_tokens=30, pad_token_id=trained_tokenizer.eos_token_id, logits_processor=[logits_processor])
            question_ids = question_ids[:, input_ids.shape[1]:]  # Remove prompt tokens
            question = trained_tokenizer.decode(question_ids[0], skip_special_tokens=True)
            question = question.split("?")[0] + "?"
            question = question.strip()
            print(f"Q{questions + 1}: {question}")
            context = oracle_tokenizer(generate_oracle_prompt(target_city, question), return_tensors="pt").to(device)
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
    """


def reward_trajectories(trajectories):
    rewards = []
    for trajectory in trajectories:
        reward = -len(trajectory["questions"])
        final_question = trajectory["questions"][-1]
        if trajectory["target_city"].lower() in final_question.lower():
            reward += 1
        rewards.append(reward)
    return rewards

def run_evaluation(gpt2MC, step):
    trajectories = generate_trajectories(gpt2MC, 20)
    rewards = reward_trajectories(trajectories)
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    accuracy = np.sum([t['target_city'].lower() in t['questions'][-1].lower() for t in trajectories]) / len(trajectories)
    print(f"Average reward: {avg_reward}")
    print(f"Std reward: {std_reward}")
    print(f"Accuracy: {accuracy}")

    with open(f"mc/summary_mc_{step}.out", "w") as f:
        f.write(f"Average reward: {avg_reward}\n")
        f.write(f"Std reward: {std_reward}\n")
        f.write(f"Accuracy: {accuracy}\n")

    with open(f"mc/trajectories_mc_{step}.out", "w") as f:
        json.dump(trajectories, f, indent=2)
    


if __name__ == "__main__":
    
    with open('train.json', 'r') as f:
        conversations = json.load(f)
    count = 0
    max_conversations = 20_000

    base_model_path = "./gpt2-medium-offline"
    sft_model_path = "bc/fine_tuned_gpt2_medium_lora"

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    sft_model = PeftModel.from_pretrained(base_model, sft_model_path)

    gpt2MC = GPT2MC(sft_model, tokenizer)
    model = gpt2MC.model 

    #run_evaluation(gpt2MC, 0)

    run = wandb.init(
        project="research-multi-turn-RL-with-LMs",
        name="gpt2-medium-mc-lora",
        config={
            "learning_rate": 1e-4,       # smaller LR for stability
            "batch_size": 4,
            "max_length": tokenizer.model_max_length,
            "epochs": 2,
            "optimizer": "AdamW",
            "accum_steps": 8,            # gradient accumulation
        },
    )

    dataset = MCDataset(conversations, tokenizer, max_conversations=max_conversations)    
    dataloader = DataLoader(dataset, batch_size=run.config['batch_size'], shuffle=True) #, collate_fn=lambda b: collate_fn(b, tokenizer))

    total_steps = len(dataloader) * run.config["epochs"]
    trainable_params = [p for p in gpt2MC.q_function.parameters()] # Q-Function
    optimizer = optim.Adam(trainable_params, lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.05 * total_steps, num_training_steps=total_steps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt2MC.to(device)
    gpt2MC.train()





    for epoch in tqdm.tqdm(range(run.config["epochs"]), desc="Epochs"):
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            should_take_action = batch['should_take_action'].to(device)
            returns = batch['returns'].to(device)

            logits, q_values = gpt2MC(input_ids, attention_mask)

            loss, logs = mc_loss(q_values, logits, input_ids, attention_mask, should_take_action, returns)

            if (step + 1) % 100 == 0:
                wandb.log({
                    "total_loss": loss.item(), 
                    "q_loss": logs['q_loss'],
                    "q_cql_loss": logs['q_cql_loss'],
                    "epoch": epoch,
                    "step": step + epoch * len(dataloader)
                })

            loss /= run.config["accum_steps"]
            loss.backward()

            if (step + 1) % run.config["accum_steps"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            #20_000 / 4 = 5_000 batches
            # Every 1250 batches run evaluation (should be 2 (epochs) x 4 (ev/epoch) = 8)
            if (step + 1) % (len(dataloader) / 4) == 0:
                run_evaluation(gpt2MC, step+1)

    torch.save(gpt2MC.q_function.state_dict(), "mc/q_function.pth")

            
         
        
    