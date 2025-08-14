import torch
import math
import torch.nn as nn

class InputEmbeddings(nn.module): 
    # vocab_size - number of words, d_model - number of features in a word
    def __init__(self, vocab_size, d_model): 
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncodings(nn.module): 
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
    
