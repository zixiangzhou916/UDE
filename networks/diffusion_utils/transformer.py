import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import math
import importlib

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        if self.batch_first:
            x = x + self.pe.permute(1, 0, 2)[:, :x.shape[1], :]
        else:
            x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class StylizationBlock(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h

class LinearTemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, T, D
        key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, T, H, HD
        value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(xf))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(xf)).view(B, N, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class FFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, ffn_dim)
        self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    def forward(self, x, emb):
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        y = x + self.proj_out(y, emb)
        return y
    
class LinearTemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 cond_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.sa_block = LinearTemporalSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = LinearTemporalCrossAttention(
            seq_len, latent_dim, cond_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask):
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb)
        x = self.ffn(x, emb)
        return x

class TemporalSelfAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(latent_dim, latent_dim)
        self.value = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, emb, src_mask):
        """
        x: B, T, D
        """
        B, T, D = x.shape
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, T, D
        key = self.key(self.norm(x)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, T, H, -1)
        # B, T, T, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        attention = attention + (1 - src_mask.unsqueeze(-1)) * -100000
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.norm(x)).view(B, T, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class TemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, cond_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.cond_norm = nn.LayerNorm(cond_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(cond_latent_dim, latent_dim)
        self.value = nn.Linear(cond_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
    
    def forward(self, x, xf, emb):
        """
        x: B, T, D
        xf: B, N, L
        """
        B, T, D = x.shape
        N = xf.shape[1]
        H = self.num_head
        # B, T, 1, D
        query = self.query(self.norm(x)).unsqueeze(2)
        # B, 1, N, D
        key = self.key(self.cond_norm(xf)).unsqueeze(1)
        query = query.view(B, T, H, -1)
        key = key.view(B, N, H, -1)
        # B, T, N, H
        attention = torch.einsum('bnhd,bmhd->bnmh', query, key) / math.sqrt(D // H)
        weight = self.dropout(F.softmax(attention, dim=2))
        value = self.value(self.cond_norm(xf)).view(B, N, H, -1)
        y = torch.einsum('bnmh,bmhd->bnhd', weight, value).reshape(B, T, D)
        y = x + self.proj_out(y, emb)
        return y

class TemporalDiffusionTransformerDecoderLayer(nn.Module):

    def __init__(self,
                 seq_len=60,
                 latent_dim=32,
                 cond_latent_dim=512,
                 time_embed_dim=128,
                 ffn_dim=256,
                 num_head=4,
                 dropout=0.1):
        super().__init__()
        self.sa_block = TemporalSelfAttention(
            seq_len, latent_dim, num_head, dropout, time_embed_dim)
        self.ca_block = TemporalCrossAttention(
            seq_len, latent_dim, cond_latent_dim, num_head, dropout, time_embed_dim)
        self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

    def forward(self, x, xf, emb, src_mask):
        x = self.sa_block(x, emb, src_mask)
        x = self.ca_block(x, xf, emb)
        x = self.ffn(x, emb)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
    
class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.pose_embedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.vel_embedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        """
        :param x: [batch_size, nframes, dim]
        """
        bs, nframes, dim = x.shape
        # bs, njoints, nfeats, nframes = x.shape
        # x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)
        x = x.permute(1, 0, 2)

        if self.data_rep in ['rot', 'xyz', 'hml_vec']:
            x = self.pose_embedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.pose_embedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.vel_embedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError
        
class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.pose_final = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.vel_final = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        """
        :param output: [nframes, batch_size, dim]
        """
        nframes, bs, d = output.shape
        if self.data_rep in ['rot', 'xyz', 'hml_vec']:
            output = self.pose_final(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.pose_final(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.vel_final(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.permute(1, 0, 2)    # [batch_size, nframes, dim]
        return output

"""Input condition motions are discrete tokens, and we encode the condition to single embedding, 
and we adopt the architecture described in MDM: https://github.com/GuyTevet/motion-diffusion-model
""" 
class MotionTransformer(nn.Module):
    def __init__(self, 
                 input_feats,           # 263
                 num_frames=240,        # 196
                 latent_dim=512,        # 512
                 cond_latent_dim=512,   # 512
                 num_tokens=2048,       # 2048
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0,
                 activation="gelu", 
                 num_motion_layers=4, 
                 motion_latent_dim=256, 
                 motion_ff_size=2048, 
                 motion_num_heads=4, 
                 no_eff=False, 
                 decoder_arch="trans_dec", 
                 cond_mask_prob=0.0, **kwargs):
        super(MotionTransformer, self).__init__()
        assert decoder_arch in ["trans_enc", "trans_dec", "gru"]

        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation  
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim * 4
        self.decoder_arch = decoder_arch
        self.gru_emb_dim = self.latent_dim if self.decoder_arch == "gru" else 0
        self.cond_mask_prob = cond_mask_prob
        self.sequence_embedding = PositionalEncoding(d_model=latent_dim, dropout=dropout, max_len=5000)
        
        self.cond_embedding = nn.Embedding(num_embeddings=num_tokens, embedding_dim=latent_dim)
        cond_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, 
                                                        nhead=num_heads, 
                                                        dim_feedforward=ff_size, 
                                                        dropout=dropout, 
                                                        activation=activation)
        self.cond_encoder = nn.TransformerEncoder(encoder_layer=cond_encoder_layer, 
                                                  num_layers=num_layers)
        
        self.input_process = InputProcess(data_rep="rot", 
                                          input_feats=input_feats+self.gru_emb_dim, 
                                          latent_dim=latent_dim)
        
        if self.decoder_arch == "trans_enc":
            transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, 
                                                                   nhead=num_heads, 
                                                                   dim_feedforward=ff_size, 
                                                                   dropout=dropout, 
                                                                   activation=activation)
            self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, 
                                                             num_layers=num_layers)
        
        elif self.decoder_arch == "trans_dec":
            transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, 
                                                                   nhead=num_heads, 
                                                                   dim_feedforward=ff_size, 
                                                                   dropout=dropout, 
                                                                   activation=activation)
            self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, 
                                                             num_layers=num_layers)
        
        elif self.decoder_arch == "gru":
            self.gru = nn.GRU(latent_dim, latent_dim, num_layers=num_layers, batch_first=True)
        
        else:
            raise ValueError("{:s} is not recognized!")
        
        self.embed_timestep = TimestepEmbedder(latent_dim=latent_dim, sequence_pos_encoder=self.sequence_embedding)
        
        self.output_process = OutputProcess(data_rep="rot", input_feats=input_feats, latent_dim=latent_dim)
        
    def generate_src_mask(self, T, length):
        B = len(length)
        src_mask = torch.ones(B, T)
        for i in range(B):
            for j in range(length[i], T):
                src_mask[i, j] = 0
        return src_mask
    
    def encode_motion(self, motion):
        """
        :param motion: [batch_size, nframes]
        """
        emb = self.cond_embedding(motion)   # [batch_size, nframes] -> [batch_size, nframes, dim]
        emb = emb.permute(1, 0, 2)
        emb = self.sequence_embedding(emb)
        emb = self.cond_encoder(emb).permute(1, 0, 2)
        return emb
             
    def mask_cond(self, cond, force_mask=False):
        bs, t, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, t, device=cond.device) * self.cond_mask_prob).view(bs, t, 1)
            return cond * (1. - mask)
        else:
            return cond
        
    def forward(self, x, timesteps, cond=None, cond_emb=None):
        """
        :param x: [batch_size, seq_len, dim], noisy data
        :param timesteps: [batch_size]
        :param cond: [batch_size, len, dim], sequential embedding
        """
        batch_size, nframes = x.shape[:2]
        
        # Embed condition
        if cond_emb is None:
            cond_emb = self.encode_motion(motion=cond)
        cond_emb, _ = cond_emb.max(dim=1, keepdim=True)
        
        # Embed the timestep
        emb = self.embed_timestep(timesteps)    # [1, batch_size, dim]
        
        # Add condition embedding to timestep embedding
        # emb = emb * xf_out.permute(1, 0, 2)
        emb = emb + self.mask_cond(cond_emb, force_mask=False).permute(1, 0, 2)
        emb_length = emb.shape[0]
        
        if self.decoder_arch == "gru":
            x_reshape = x.permute(0, 2, 1).reshape(batch_size, -1, 1, nframes)
            emb_gru = emb.repeat(nframes, 1, 1)
            emb_gru = emb_gru.permute(1, 2, 0).reshape(batch_size, self.latent_dim, 1, nframes)
            x = torch.cat((x_reshape, emb_gru), dim=1)
            
        x = self.input_process(x)   # [batch_size, seq_len, 75] -> [seq_len, batch_size, dim_latent]
        # src_mask = self.generate_src_mask(T=nframes, length=length).to(x.device).bool()
        
        if self.decoder_arch == "trans_enc":
            # Adding the timestep embed
            xseq = torch.cat((emb, x), axis=0)                      # [seq_len+emb_len, batch_size, dim]
            xseq = self.sequence_embedding(xseq)                    # [seq_len+emb_len, batch_size, dim]
            output = self.transformer_encoder(xseq)[emb_length:]    # [seq_len, batch_size, dim]
            
        elif self.decoder_arch == "trans_dec":
            xseq = self.sequence_embedding(x)
            output = self.transformer_encoder(tgt=xseq, memory=emb)
        
        elif self.decoder_arch == "gru":
            xseq = self.sequence_embedding(x)
            output, _ = self.gru(xseq)
            
        output = self.output_process(output)    # [batch_size, nframes, dim]
        return output
             

        
if __name__ == "__main__":
    pass