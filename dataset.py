from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# MSRP is a dataset that contains labels for whether sentence 2 is a paraphrase
# for sentence 1, however, I want this to be a generation task, not a classification task,
# so I'm only using the examples where the label is 1, where the sentences are paraphrases

# function to clean the raw data and convert it into a tuple 
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

def collate_fn(batch):
    s1, s2 = zip(*batch)
    inputs = tokenizer(list(s1), padding=True, truncation=True, return_tensors="pt")
    labels = tokenizer(list(s2), padding=True, truncation=True, return_tensors="pt")["input_ids"]
    inputs["labels"] = labels
    return inputs

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

