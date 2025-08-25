import torch
from dataset import tokenizer
from transformer import Transformer, pad_mask, causal_mask

# Hyperparams - make sure they match training 
vocab_size = len(tokenizer)
d_model = 128
num_layers = 3
num_heads = 4
d_ff = 512
max_seq_len = 128
dropout = 0.1

# Load trained model
model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout)
model.load_state_dict(torch.load("transformer.pth"))
model.eval()

def generate_text(prompt, max_len=20):
    # Encode source (prompt sentence)
    src = tokenizer.encode(prompt, add_special_tokens=False)
    src = torch.tensor([src])   # shape [1, seq_len]

    # Decoder starts with <sos>
    decoder_input = torch.tensor([[tokenizer.bos_token_id]])

    for _ in range(max_len):
        # build masks
        src_mask = pad_mask(src, tokenizer.pad_token_id, target_len=src.size(1))
        tgt_mask = causal_mask(decoder_input.size(1)) & pad_mask(
            decoder_input, tokenizer.pad_token_id, target_len=decoder_input.size(1)
        )
        cross_mask = pad_mask(src, tokenizer.pad_token_id, target_len=decoder_input.size(1))

        # forward pass
        output = model(src, decoder_input, src_mask, tgt_mask, cross_mask)

        # take the last predicted token
        next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(0)

        # append it to decoder input
        decoder_input = torch.cat([decoder_input, next_token], dim=1)

        # stop if EOS is predicted
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode tokens into text (skip <sos> and <pad>)
    tokens = decoder_input.squeeze().tolist()
    return tokenizer.decode(tokens, skip_special_tokens=True)


print(generate_text("Hello World!", max_len=30))