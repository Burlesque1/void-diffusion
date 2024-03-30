import torch
import torch.nn as nn
import math

if torch.cuda.is_available():
  device = torch.device('cuda')
elif torch.backends.mps.is_available():
  device = torch.device('mps')
else:
  device = torch.device('cpu')

# device = torch.device('cpu')

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) # (batch_size, num_heads, seq_length, seq_length)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        x.view(batch_size, seq_length, self.num_heads, self.d_k) # (batch_size, seq_length, num_heads, d_k)
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, seq_length, d_k)  
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) # (batch_size, seq_length, d_model)
        
    def forward(self, Q, K, V, mask=None):
        self.W_q(Q) # (batch_size, seq_length, d_model)
        Q = self.split_heads(self.W_q(Q)) # (batch_size, num_heads, seq_length, d_k)
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask) # (batch_size, num_heads, seq_length, d_k)
        output = self.W_o(self.combine_heads(attn_output)) # (batch_size, seq_length, d_model)
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x))) # no dropout
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model) # 100 * 512
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) # 100 * 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) # 1 * 256
        
        pe[:, 0::2] = torch.sin(position * div_term) # Starting from 0 with a gap of two
        pe[:, 1::2] = torch.cos(position * div_term) # Starting from 1 with a gap of two
        
        self.register_buffer('pe', pe.unsqueeze(0)) # 1 * 100 * 512
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)] # [row : column]  

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output)) # (batch_size, seq_len, d_model)
        return x
    
# nn.TransformerDecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model) # 5000 * 512
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]) 
        # nn.TransformerDecoder
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) # 64 * 1 * 1 * 100
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3) # 64 * 1 * 100 * 1
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(device) # 1 * 100 * 100
        print(src_mask.device,tgt_mask.device,nopeak_mask.device)
        tgt_mask = tgt_mask & nopeak_mask # 64 * 1 * 100 * 100
        return src_mask, tgt_mask # [64, 1, 1, 100] [64, 1, 100, 100]

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt) # [64, 1, 1, 100] [64, 1, 100, 100]
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src))) # [64, 100, 512]
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt))) # [64, 100, 512]

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output) # [64, 100, 5000]
        return output