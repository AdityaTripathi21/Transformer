import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class InputEmbedding(nn.Module): 
    # vocab_size - number of words, d_model - number of features in a word
    def __init__(self, vocab_size, d_model): 
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module): 
    def __init__(self, d_model, max_seq_length): 
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # div_term - a tensor to divide the position tensor by
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))

        # sin for even columns, cos for odd columns
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # store pe as untrainable parameter and unsqueeze is for adding batch dimension for addition
        # pe is added automatically to other batches, it's "broadcasted" in a sense
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads # - use // for integer floor

        # Projection layers of Q, K, and V - Wq, Wk, Wv, shape of (d_model, d_model)
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)

        self.output_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        seq_length = x.size(1)
        x = x.reshape(batch_size, seq_length, self.num_heads, self.head_dim) # adds extra dims
        return x.permute(0,2,1,3) # reoders dims
    
    def compute_attention(self, query, key, value, mask=None): 
        # attention formula - dot product of Q and K divided by square root of head dims + softmax
        scores = torch.matmul(query, key.transpose(-2,-1)) / (self.head_dim ** 0.5)
        if mask is not None: 
            scores = scores.masked_fill(mask==0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, value)
    
    def combine_heads(self, x, batch_size): 
        seq_length = x.size(2)
        x = x.permute(0,2,1,3).contiguous()
        return x.view(batch_size, seq_length, self.d_model) # shape - (B, T, d_model)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # where x gets multiplied by Wq, Wk, Wv (they are the weight matrices)
        query = self.split_heads(self.query_linear(query), batch_size)
        key = self.split_heads(self.key_linear(key), batch_size)
        value = self.split_heads(self.value_linear(value), batch_size)

        attention_weights = self.compute_attention(query, key, value, mask)

        output = self.combine_heads(attention_weights, batch_size)
        return self.output_linear(output)


class FeedForwardSublayer (nn.Module): 
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        return self.net(x)

