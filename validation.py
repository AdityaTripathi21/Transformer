import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader
from dataset import ParaphraseDataset, collate_fn, load_test_data, tokenizer
from transformer import Transformer, pad_mask, causal_mask

vocab_size = len(tokenizer)
d_model = 128
num_layers = 3
num_heads = 4
d_ff = 512
max_seq_len = 128
dropout = 0.1

model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout)
model.load_state_dict(torch.load("transformer.pth"))
model.eval()

test_data = load_test_data()
test_dataset = ParaphraseDataset(test_data)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

def evaluate(model, dataloader):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    total_loss, total_correct, total_tokens = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch["input_ids"]
            tgt = batch["labels"]

            decoder_input = tgt[:, :-1]
            labels = tgt[:, 1:]

            src_mask = pad_mask(src, tokenizer.pad_token_id, src.size(1))
            tgt_mask = causal_mask(decoder_input.size(1))

            # forward
            output = model(src, decoder_input, src_mask, tgt_mask, src_mask)

            # loss
            loss = criterion(output.reshape(-1, output.size(-1)), labels.reshape(-1))
            total_loss += loss.item()

            # accuracy
            preds = output.argmax(dim=-1)
            mask = labels != tokenizer.pad_token_id
            total_correct += ((preds == labels) & mask).sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, accuracy, perplexity

# run eval
if __name__ == "__main__":
    loss, acc, ppl = evaluate(model, test_dataloader)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test Perplexity: {ppl:.2f}")
