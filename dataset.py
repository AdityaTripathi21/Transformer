from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch 

# MSRP is a dataset that contains labels for whether sentence 2 is a paraphrase
# for sentence 1, however, I want this to be a generation task, not a classification task,
# so I'm only using the examples where the label is 1, where the sentences are paraphrases

# function to clean the raw training data and convert it into a tuple 
# tuple format - (sentence, sentence)
# bidirectional to double the small dataset
def load_training_data(bidirectional=True):
    file = "msr_paraphrase_train.txt"
    dataset = []
    with open(file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.lower().strip().split("\t")
            label = int(parts[0])
            s1 = parts[3].strip()
            s2 = parts[4].strip()
            if label == 1:
                dataset.append((s1, s2))
                if bidirectional:
                    dataset.append((s2, s1))

    return dataset

# function to clean the raw test data and convert it into a tuple 
# would've been better to make one function and take in txt file as argument
# however, I decided to do validation a lot later
def load_test_data(bidirectional=True):
    file = "msr_paraphrase_test.txt"
    dataset = []
    with open(file, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.lower().strip().split("\t")
            label = int(parts[0])
            s1 = parts[3].strip()
            s2 = parts[4].strip()
            if label == 1:
                dataset.append((s1, s2))
                if bidirectional:
                    dataset.append((s2, s1))

    return dataset

# Custom Dataset implementation must have init, len, and getitem, need Dataset for DataLoader
class ParaphraseDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        s1, s2 = self.data[index]
        return s1, s2


data = load_training_data()
dataset = ParaphraseDataset(data)

tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.add_special_tokens({
    "bos_token": "<sos>",
    "eos_token": "<eos>"
})

# tokenize input and also add padding and sos and eos and return dictionary
def collate_fn(batch):
    s1, s2 = zip(*batch)
    inputs = tokenizer(list(s1), padding=True, return_tensors="pt")

    labels = []
    for sent in s2:
        tokens = tokenizer.encode(
            sent,
            add_special_tokens=False  
        )

        tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]

        labels.append(tokens)

    max_len = max(len(l) for l in labels)
    labels = [
    l + [tokenizer.pad_token_id] * (max_len - len(l)) for l in labels
    ]

    inputs["labels"] = torch.tensor(labels)
    return inputs

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

