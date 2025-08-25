from dataset import dataloader, tokenizer
from transformer import Transformer, pad_mask, causal_mask
import torch
import torch.nn as nn
import torch.optim as optim

# hyperparams
vocab_size = len(tokenizer)
d_model = 128
num_layers = 3
num_heads = 4
d_ff = 512
max_seq_len = 128
dropout = 0.1

num_epochs = 25

model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout)
# use CEL because multi-class classifcation
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

optimizer = optim.Adam(model.parameters(), lr=5e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        src = batch["input_ids"]
        tgt = batch["labels"]

        decoder_input = tgt[:, :-1]
        labels = tgt[:, 1:]

        # masks
        src_mask = pad_mask(src, tokenizer.pad_token_id, target_len=src.size(1))
        tgt_mask = causal_mask(decoder_input.size(1)) & pad_mask(decoder_input, tokenizer.pad_token_id, target_len=decoder_input.size(1))
        cross_mask = pad_mask(src, tokenizer.pad_token_id, target_len=decoder_input.size(1))  
  

        # forward
        output = model(src, decoder_input, src_mask, tgt_mask, cross_mask)

        # loss expects [N, vocab_size] and the target is just [N] where N is the number of tokens
        # so we need .view to resize the output 
        loss = criterion(output.reshape(-1, output.size(-1)), labels.reshape(-1))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), "transformer.pth")
print("Model saved to transformer.pth")
