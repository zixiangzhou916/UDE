import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer('positional_encoding', self.encoding)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        return self.encoding[:seq_len, :].clone().detach().to(x.device) + x
    
def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            # out = fused_leaky_relu(out, self.bias * self.lr_mul)  # DEBUG!!!

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )
        
class ScaledStylizationBlock(nn.Module):
    def __init__(self, emb_dim, latent_dim, dropout):
        super(ScaledStylizationBlock, self).__init__()
        self.latent_emb_layers = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(latent_dim, 2 * emb_dim)
        )
        self.emb_norm = nn.LayerNorm(emb_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(), 
            nn.Dropout(p=dropout), 
            nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, emb, latent):
        """
        :param emb: [batch_size, nframes, dim]
        :param latent: [batch_size, dim]
        """
        lat_out = self.latent_emb_layers(latent)    # [bs, 1, dim] -> [bs, 1, 2*dim]
        scale, shift = torch.chunk(lat_out, 2, dim=2)
        h = self.emb_norm(emb) * (1 + scale) + shift
        h = self.out_layers(h)
        return h
    
class AttentiveStylizationBlock(nn.Module):
    def __init__(self, emb_dim, latent_dim, dropout):
        """
        :param emb_dim: dimension of query
        :param latent_dim: dimension of key
        """
        super(AttentiveStylizationBlock, self).__init__()
        self.q_layer = nn.Linear(emb_dim, emb_dim)
        self.k_layer = nn.Linear(latent_dim, emb_dim)
        self.v_layer = nn.Linear(latent_dim, emb_dim)
        self.softmax = nn.Softmax(dim=1)
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.dim = emb_dim
        
        self.q_layer.apply(init_weight)
        self.k_layer.apply(init_weight)
        self.v_layer.apply(init_weight)
    
    def forward(self, emb, latent):
        """
        :param emb: [batch_size, nframes, dim], query tensor
        :param latent: [batch_size, dim], key and value tensor
        """
        query_tensor = self.q_layer(emb)                    # [bs, seq_len, dim]
        val_tensor = self.v_layer(latent)                   # [bs, 1, dim]  
        key_tensor = self.k_layer(latent)                   # [bs, 1, dim]
        
        weights = torch.matmul(query_tensor, key_tensor.transpose(1, 2)) / np.sqrt(self.dim)    # [bs, seq_len, 1]
        weights = self.softmax(weights)  # [bs, seq_len, 1]
        
        pred = torch.matmul(weights, val_tensor)    # [bs, seq_len, dim]
        
        return self.layer_norm(pred + emb)
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, d_latent, dropout=0.1, style_module="scaled"):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        if style_module == "scaled":
            self.proj_out = ScaledStylizationBlock(emb_dim=d_model, latent_dim=d_latent, dropout=dropout)
        elif style_module == "attn":
            self.proj_out = AttentiveStylizationBlock(emb_dim=d_model, latent_dim=d_latent, dropout=dropout)

    def forward(self, q, k, v, latent=None, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))    # [batch_size, nframes, dim]
        
        if latent is not None:
            q = self.proj_out(emb=q, latent=latent)
            
        q += residual
        q = self.layer_norm(q)
        
        return q, attn
    
class FFN(nn.Module):
    
    def __init__(self, emb_dim, latent_dim, ffn_dim, dropout, style_module="scaled"):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(emb_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, emb_dim)
        self.layer_norm = nn.LayerNorm(emb_dim, eps=1e-6)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if style_module == "scaled":
            self.proj_out = ScaledStylizationBlock(emb_dim=emb_dim, latent_dim=latent_dim, dropout=dropout)
        elif style_module == "attn":
            self.proj_out = AttentiveStylizationBlock(emb_dim=emb_dim, latent_dim=latent_dim, dropout=dropout)

    def forward(self, x, emb=None):
        
        residual = x
        x = self.linear2(self.activation(self.linear1(x)))
        x = self.dropout(x)
        
        if emb is None:
            y = residual + x
        else:
            y = residual + self.proj_out(x, emb)
        
        y = self.layer_norm(y)
        
        # y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # y = x + self.proj_out(y, emb)
        return y

class AttLayer(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(AttLayer, self).__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)

        self.softmax = nn.Softmax(dim=1)
        self.dim = value_dim

        self.W_q.apply(init_weight)
        self.W_k.apply(init_weight)
        self.W_v.apply(init_weight)

    def forward(self, query, key_mat):
        '''
        query (batch, query_dim)
        key (batch, seq_len, key_dim)
        '''
        # print(query.shape)
        query_vec = self.W_q(query).unsqueeze(-1)       # (batch, value_dim, 1)
        val_set = self.W_v(key_mat)                     # (batch, seq_len, value_dim)
        key_set = self.W_k(key_mat)                     # (batch, seq_len, value_dim)

        weights = torch.matmul(key_set, query_vec) / np.sqrt(self.dim)

        co_weights = self.softmax(weights)              # (batch, seq_len, 1)
        values = val_set * co_weights                   # (batch, seq_len, value_dim)
        pred = values.sum(dim=1)                        # (batch, value_dim)
        return pred, co_weights
 
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, d_latent, n_head, d_k, d_v, dropout=0.1, style_module="naive"):
        super(DecoderLayer, self).__init__()
        assert style_module in ["scaled", "attn"]
        
        self.slf_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, 
                                           d_k=d_k, d_v=d_v, 
                                           d_latent=d_latent, 
                                           dropout=dropout, 
                                           style_module=style_module)
        self.crs_attn = MultiHeadAttention(n_head=n_head, d_model=d_model, 
                                           d_k=d_k, d_v=d_v, 
                                           d_latent=d_latent, 
                                           dropout=dropout, 
                                           style_module=style_module)
        self.ffn = FFN(emb_dim=d_model, latent_dim=d_latent, ffn_dim=d_inner, 
                       dropout=dropout, style_module=style_module)
        
    def forward(self, dec_input, enc_output, latent=None, slf_attn_mask=None, crs_attn_mask=None):
        
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, latent, mask=slf_attn_mask)
        dec_output, dec_crs_attn = self.crs_attn(dec_output, enc_output, enc_output, latent, mask=crs_attn_mask)
        dec_output = self.ffn(dec_output, latent)
        
        return dec_output, dec_slf_attn, dec_crs_attn
    
class MotionTransformerV1(nn.Module):
    def __init__(self, d_model, d_latent, d_inner, d_k, d_v, 
                 n_head, n_layers, n_tokens, 
                 dropout, pad_idx, style_module, **kwargs):
        super(MotionTransformerV1, self).__init__()
        assert style_module in ["scaled", "attn"]
        self.position_enc = PositionalEncoding(d_model=d_model, max_len=5000)
        self.token_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model, padding_idx=pad_idx)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = DecoderLayer(d_model=d_model, d_inner=d_inner, d_latent=d_latent, 
                                 n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                                 style_module=style_module)
            self.layers.append(layer)
        self.d_model = d_model
        
    def forward(self, trg_seq, trg_mask, enc_output, src_mask, latent, return_attns=False):
        """
        :param trg_seq:
        :param trg_mask: [batch_size, trg_seq_len-1, trg_seq_len-1]
        :param enc_output: [batch_size, 77, emb_dim]
        :param src_mask: [batch_size, 1, 77]
        """
        dec_slf_attn_list, dec_crs_attn_list = [], []
        
        dec_emb = self.token_emb(trg_seq)
        dec_emb = (self.d_model ** 0.5) * dec_emb
        
        dec_emb = self.position_enc(dec_emb)
        
        for layer in self.layers:
            dec_emb, dec_slf_attn, dec_crs_attn = layer(dec_emb, enc_output, latent, 
                                                        slf_attn_mask=trg_mask, 
                                                        crs_attn_mask=src_mask)
        dec_slf_attn_list += [dec_slf_attn] if return_attns else []
        dec_crs_attn_list += [dec_crs_attn] if return_attns else []
        
        if return_attns:
            return dec_emb, dec_slf_attn_list, dec_crs_attn_list
        else:
            return dec_emb, 

"""Use TransformerDecoder."""
class MotionTransformerV2(nn.Module):
    def __init__(self, d_model, d_latent, d_inner, d_k, d_v, 
                 n_head, n_layers, n_tokens, 
                 dropout, pad_idx, style_module, **kwargs):
        super(MotionTransformerV2, self).__init__()
        assert style_module in ["scaled", "attn"]
        self.position_enc = PositionalEncoding(d_model=d_model, max_len=5000)
        self.token_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model, padding_idx=None)
        
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, 
                                                               dim_feedforward=d_inner, 
                                                               dropout=dropout, 
                                                               activation="gelu")
        self.layers = nn.TransformerDecoder(decoder_layer=transformer_decoder_layer, num_layers=n_layers)
        self.head = nn.Linear(in_features=d_model, out_features=n_tokens)
        self.d_model = d_model
        
    def forward(self, enc_output, querys):
        """
        :param enc_output: [batch_size, 77, emb_dim]
        :param querys: [batch_size, nquerys, query_dim]
        """
        query_emb = self.position_enc(querys)
        output = self.layers(tgt=query_emb.permute(1, 0, 2), 
                             memory=enc_output.permute(1, 0, 2))
        output = output.permute(1, 0, 2)
        output = self.head(output)
        return output

"""Ude TransformerDecoder. Apply mask to tgt."""
class MotionTransformerV3(nn.Module):
    def __init__(self, d_model, d_latent, d_inner, d_k, d_v, 
                 n_head, n_layers, n_tokens, 
                 dropout, pad_idx, style_module, **kwargs):
        super(MotionTransformerV3, self).__init__()
        assert style_module in ["scaled", "attn"]
        self.position_enc = PositionalEncoding(d_model=d_model, max_len=5000)
        self.learnable_pos_enc = nn.Parameter(torch.randn(200, d_model), requires_grad=True)
        self.token_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model, padding_idx=None)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), 
            nn.LayerNorm(d_model)
        )
        
        transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, 
                                                               dim_feedforward=d_inner, 
                                                               dropout=dropout, 
                                                               activation="gelu")
        self.layers = nn.TransformerDecoder(decoder_layer=transformer_decoder_layer, num_layers=n_layers)
        self.head = nn.Linear(in_features=d_model, out_features=n_tokens-1) # Make sure we dont predict <SOS>!
        self.d_model = d_model
        
    def forward(self, enc_output, dec_cond, querys):
        """
        :param enc_output: [batch_size, 77, emb_dim]
        :param dec_cond: [batch_size, ncond]
        :param querys: [batch_size, nquerys, query_dim]
        """
        batch_size = enc_output.shape[0]
        
        # Embed the dec_cond
        dec_cond_emb = self.token_emb(dec_cond)
        
        # 
        dec_input = torch.cat((dec_cond_emb, querys), dim=1)
        dec_input = self.position_enc(dec_input)
        
        # Learnable position encoding
        learnable_pos_enc = self.learnable_pos_enc[None, :dec_input.shape[1]].repeat(batch_size, 1, 1)
        dec_input = dec_input + learnable_pos_enc
        
        # Project
        dec_input_emb = self.proj(dec_input.permute(1, 0, 2))   # [nquerys, bs, dim]
                
        # # Get tgt mask
        # T = dec_input.shape[1]
        # cond_len = dec_cond.shape[1]
        # query_len = querys.shape[1]
        # tgt_mask = torch.zeros(T, T).to(enc_output.device)
        # tgt_mask[:cond_len, :cond_len] = 1.0
        # tgt_mask[cond_len:, cond_len:] = torch.tril(torch.ones(query_len, query_len)).to(enc_output.device)
        # tgt_mask = tgt_mask.bool()
        
        # ### DEBUG
        # import numpy as np
        # mask_ = tgt_mask.detach().cpu().numpy()
        # np.savetxt("mask.txt", mask_, fmt="%d")
        
        # Decode
        output = self.layers(tgt=dec_input_emb, memory=enc_output.permute(1, 0, 2))
        output = output.permute(1, 0, 2)
        output = self.head(output[:, -querys.shape[1]:])
        return output

class MotionGRU(nn.Module):
    def __init__(self, d_model, d_latent, d_inner, d_k, d_v, 
                 n_head, n_layers, n_tokens, 
                 dropout, pad_idx, style_module, **kwargs):
        super(MotionGRU, self).__init__()
        self.n_layers = n_layers
        
        self.input_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_inner, padding_idx=pad_idx)
        self.z2init = nn.Linear(d_model, d_inner * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(d_inner, d_inner) for i in range(n_layers)])
        
        self.att_layer = AttLayer(d_inner, d_model, d_inner)
        self.att_linear = nn.Sequential(
            nn.Linear(d_inner * 2, d_inner),
            nn.LayerNorm(d_inner),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj = nn.Linear(d_inner, n_tokens, bias=False)
        
        self.input_emb.apply(init_weight)
        self.z2init.apply(init_weight)
        # self.output_net.apply(init_weight)
        self.att_layer.apply(init_weight)
        self.att_linear.apply(init_weight)
        self.hidden_size = d_inner

        self.proj.weight = self.input_emb.weight
        
    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = torch.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)
    
    def forward(self, src_output, inputs, hidden):
        
        h_in = self.input_emb(inputs)
        att_vec, _ = self.att_layer(hidden[-1], src_output)
        # print(att_vec.shape, h_in.shape)
        h_in = self.att_linear(
            torch.cat([att_vec, h_in], dim=-1)
        )
        for i in range(self.n_layers):
            # print(h_in.shape, hidden[i].shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]

        # print(h_in.shape, src_output.shape)
        pred_probs = self.proj(h_in)
        return pred_probs, hidden
       
class ConditionEncoderV1(nn.Module):
    def __init__(self, n_tokens, d_audio, d_model, d_inner, 
                 n_head, n_layers, dropout=0.1, **kwargs):
        """
        :param n_tokens: number of motion tokens
        :param d_audio: input dimention of audio features
        :param n_layers: number of cross modal transformer layers
        :param n_head: number of heads of self-attention
        :param d_model: dimension of input to transformer encoder
        :param d_inner: dimension of intermediate layer of transformer encoder
        :param dropout: dropout rate
        :param **kwrags: any other possible arguments (optional)
        """
        super(ConditionEncoderV1, self).__init__()
        
        self.position_enc = PositionalEncoding(d_model, max_len=5000)
        self.codebook_emb = nn.Embedding(n_tokens, d_model)
        self.n_tokens = n_tokens
        
        self.a_proj = nn.Linear(d_audio, d_model, bias=False)
        self.t_proj = nn.Linear(d_model, d_model, bias=False)
        self.m_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.a_token = nn.Parameter(torch.randn(d_model), requires_grad=True)
        self.m_token = nn.Parameter(torch.randn(d_model), requires_grad=True)
        self.t_token = nn.Parameter(torch.randn(d_model), requires_grad=True)
        
        self.agg_t_token = nn.Parameter(torch.randn(d_model), requires_grad=True)
        self.agg_a_token = nn.Parameter(torch.randn(d_model), requires_grad=True)

        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                       nhead=n_head, 
                                                       dim_feedforward=d_inner, 
                                                       dropout=dropout, 
                                                       activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, 
                                                 num_layers=n_layers)
        
        self.codebook_emb.apply(init_weight)
        
    def forward(self, t_seq, a_seq, m_seq, t_mask, a_mask, m_mask):
        """
        :param t_seq: the text sequence
        :param a_seq: the audio sequence
        :param m_seq: the motion sequence
        :param t_mask: the mask of text sequence
        :param a_mask: the mask of audio sequence
        :param m_mask: the mask of mask sequence
        """
        assert (t_seq is not None) or (a_seq is not None or m_seq is not None)

        if t_seq is not None and t_mask is not None:
            return self.encode_text(t_seq=t_seq, m_seq=m_seq, t_mask=t_mask, m_mask=m_mask)
        elif a_seq is not None and m_seq is not None and a_mask is not None and m_mask is not None:
            return self.encode_audio(t_seq=t_seq, m_seq=m_seq, t_mask=t_mask, m_mask=m_mask)

    def encode_text(self, t_seq, m_seq, t_mask, m_mask):
        """
        :param t_seq: [batch_size, nframes, dim], the text sequence
        :param m_seq: (optional) [batch_size, nframes_motion, dim], the motion codebook sequence
        :param t_mask: [batch_size, nframes], the mask of text sequence
        :param m_mask: (optional) [batch_size, nframes_motion], the mask of motion codebook sequence
        """
        t_seq = self.t_proj(t_seq)
        if m_seq is not None:
            m_seq = self.codebook_emb(m_seq)
            m_seq = self.m_proj(m_seq)
            
        if m_seq is not None:
            m_token = self.m_token[None, None].repeat(m_seq.shape[0], m_seq.shape[1], 1)

        agg_token = self.agg_t_token[None, None].repeat(t_seq.shape[0], 1, 1)
        t_token = self.t_token[None, None].repeat(t_seq.shape[0], t_seq.shape[1], 1)
        token_mask = torch.ones(t_seq.shape[0], 1).bool().to(t_seq.device)

        if m_seq is not None:
            xseq = torch.cat((agg_token, m_seq + m_token, t_seq + t_token), dim=1)
        else:
            xseq = torch.cat((agg_token, t_seq + t_token), dim=1)
        
        xseq = self.position_enc(xseq)
        xseq = xseq.permute(1, 0, 2)    # [nframes+1, bs, dim]

        if m_seq is not None:
            xmask = torch.cat((token_mask, m_mask, t_mask), dim=1)  # [bs, nframes_audio+nframes_motion+1]
        else:
            xmask = torch.cat((token_mask, t_mask), dim=1)
        
        xenc = self.transformer(xseq, src_key_padding_mask=~xmask)
        xenc = xenc.permute(1, 0, 2)    # [bs, nframes+1, dim]
        glob_emb, seq_emb = xenc[:, :1], xenc[:, 1:]
        
        return glob_emb, seq_emb

    def encode_audio(self, a_seq, m_seq, a_mask, m_mask):
        """
        :param a_seq: [batch_size, nframes_audio, dim], the audio sequence
        :param m_seq: (optional) [batch_size, nframes_motion, dim], the motion codebook sequence
        :param a_mask: [batch_size, nframes_audio], the mask of audio sequence
        :param m_mask: (optional) [batch_size, nframes_motion], the mask of motion codebook sequence
        """
        a_seq = self.a_proj(a_seq)
        if m_seq is not None:
            m_seq = self.codebook_emb(m_seq)
            m_seq = self.m_proj(m_seq)

        a_token = self.a_token[None, None].repeat(a_seq.shape[0], a_seq.shape[1], 1)
        if m_seq is not None:
            m_token = self.m_token[None, None].repeat(m_seq.shape[0], m_seq.shape[1], 1)
        agg_token = self.agg_a_token[None, None].repeat(a_seq.shape[0], 1, 1)
        token_mask = torch.ones(a_seq.shape[0], 1).bool().to(a_seq.device)

        if m_seq is not None:
            xseq = torch.cat((agg_token, m_seq + m_token, a_seq + a_token), dim=1)
        else:
            xseq = torch.cat((agg_token, a_seq + a_token), dim=1)
        xseq = self.position_enc(xseq)
        xseq = xseq.permute(1, 0, 2)    # [nframes_audio+nframes_motion+1, bs, dim]

        if m_seq is not None:
            xmask = torch.cat((token_mask, m_mask, a_mask), dim=1)  # [bs, nframes_audio+nframes_motion+1]
        else:
            xmask = torch.cat((token_mask, a_mask), dim=1)

        xenc = self.transformer(xseq, src_key_padding_mask=~xmask)
        xenc = xenc.permute(1, 0, 2)    # [bs, nframes_audio+nframes_motion+1, dim]
        glob_emb, seq_emb = xenc[:, :1], xenc[:, 1:]

        return glob_emb, seq_emb
    
if __name__ == '__main__':
    pass