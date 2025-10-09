from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=1024):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        self.tokenizer.truncation_side='left'

        full = context + " " + target

        encoding = self.tokenizer(
            full,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = encoding['input_ids'].clone()
        
        target_encoding = self.tokenizer(
            target,
            max_length=self.max_length,
            padding=False,  
            truncation=True,
            return_tensors="pt",
        )
        target_length = target_encoding['input_ids'].shape[1] # Get the token length of the prompt
        
        #Mask
        #labels[:, :prompt_length] = -100
        labels[:, :-(target_length + (encoding["attention_mask"] == 0).sum().item())] = -100
        labels[encoding["attention_mask"] == 0] = -100
        
        self.tokenizer.truncation_side='right'
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }
