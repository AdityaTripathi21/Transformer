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


class FeedForwardSublayer(nn.Module): 
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSublayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # mask - mask added for padded tokens because input can be variable length
    def forward(self, x, mask): 
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff_sublayer(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSublayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    # mask - mask added for padded tokens because input can be variable length
    def forward(self, x, y, mask, cross_mask): 
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        cross_attn_output = self.cross_attn(x, y, y, cross_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        ff_output = self.ff_sublayer(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList(
            # list comprehension
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for layer in range(num_layers)]
        )

    def forward(self, x, mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout, max_seq_length):
        super().__init__()
        self.embedding = InputEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.layers = nn.ModuleList(
            # list comprehension
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for layer in range(num_layers)]
        )
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask, cross_mask):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask, cross_mask)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
    
class ClassificationHead(nn.Module):
    # num_classes - number of possible classes
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x): 
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)
    
class RegressionHead(nn.Module):
    # num_classes - number of possible classes
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x): 
        return self.fc(x)
    

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff, max_seq_len, dropout):
        super().__init__()

        self.encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers,
                                          d_ff, dropout, max_seq_len)

        self.decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_layers,
                                          d_ff, dropout, max_seq_len)
        
    def forward(self, x, src_mask, mask, cross_mask):
        encoder_output = self.encoder(x, src_mask)
        decoder_output = self.decoder(x, encoder_output, mask, cross_mask)
        return decoder_output

