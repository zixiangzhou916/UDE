# import os
# import sys
# sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from .layers import *
from .position_encoding import PositionalEncoding

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

def lengths_to_mask(lengths, device):
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

""" Conditional Discriminator.
1. Inputs are motion sequence and condition embedding.
"""
class DiscriminatorV1(nn.Module):
    """ 
    ViT-like discriminator
    """
    def __init__(self, input_dim, cond_dim, channels, n_down, num_heads, hidden_dim, num_layers, dropout, activation, **kwargs):
        super(DiscriminatorV1, self).__init__()
        assert len(channels) == n_down
        
        # Convolution layers
        layers = [
            nn.Conv1d(input_dim, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.conv_layers = nn.Sequential(*layers)
        
        self.cond_linear = nn.Linear(cond_dim, channels[-1], bias=False)
        
        # Transformer layers
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=channels[-1],
                                                               nhead=num_heads,
                                                               dim_feedforward=hidden_dim,
                                                               dropout=dropout,
                                                               activation=activation)
        self.tfm_layers = nn.TransformerEncoder(transformer_encoder_layer, 
                                                num_layers=num_layers)
        self.sequence_pose_encoding = PositionalEncoding(channels[-1], dropout=dropout)
        self.final = nn.Linear(channels[-1], 1)
        
    def forward(self, motion, cond_emb):
        """
        :param motion: [batch_size, num_frames, dim]
        :param cond_emb: [batch_size, 1, dim]
        """
        lengths = [x.shape[0] for x in motion]
        cond_x = self.cond_linear(cond_emb)
        cond_x = cond_x.permute(1, 0, 2)    # [1, batch_size, dim]
        
        x = self.conv_layers(motion.permute(0, 2, 1))
        x = x.permute(2, 0, 1)  # [npatches, batch_size, dim]
        
        xseq = self.sequence_pose_encoding(x)
        xseq = self.tfm_layers(xseq + cond_x).permute(1, 0, 2)
        
        validity = self.final(xseq)
        
        return validity
        
class DiscriminatorV2(nn.Module):
    """ 
    ViT-like discriminator
    """
    def __init__(self, input_dim, channels, n_down, num_heads, hidden_dim, num_layers, dropout, activation, **kwargs):
        super(DiscriminatorV2, self).__init__()
        assert len(channels) == n_down
        
        # Convolution layers
        layers = [
            nn.Conv1d(input_dim, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.conv_layers = nn.Sequential(*layers)
                
        # Transformer layers
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=channels[-1],
                                                               nhead=num_heads,
                                                               dim_feedforward=hidden_dim,
                                                               dropout=dropout,
                                                               activation=activation)
        self.tfm_layers = nn.TransformerEncoder(transformer_encoder_layer, 
                                                num_layers=num_layers)
        self.sequence_pose_encoding = PositionalEncoding(channels[-1], dropout=dropout)
        self.final = nn.Linear(channels[-1], 1)
        
    def forward(self, motion):
        """
        :param motion: [batch_size, num_frames, dim]
        """
        lengths = [x.shape[0] for x in motion]
        
        x = self.conv_layers(motion.permute(0, 2, 1))
        x = x.permute(2, 0, 1)  # [npatches, batch_size, dim]
        
        xseq = self.sequence_pose_encoding(x)
        xseq = self.tfm_layers(xseq).permute(1, 0, 2)
        
        validity = self.final(xseq)
        
        return validity

class DiscriminatorV3(nn.Module):
    """ 
    1D-CNN discriminator
    """
    def __init__(self, input_dim, channels, n_down, num_heads, hidden_dim, num_layers, dropout, activation, **kwargs):
        super(DiscriminatorV3, self).__init__()
        assert len(channels) == n_down
        
        # Convolution layers
        layers = [
            nn.Conv1d(input_dim, channels[0], 4, 2, 1),
            nn.BatchNorm1d(channels[0]), 
            nn.LeakyReLU(0.2, inplace=True)
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.BatchNorm1d(channels[i]), 
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.conv_layers = nn.Sequential(*layers)
        self.final = nn.Conv1d(channels[-1], 1, 3, 1, 1)
        
    def forward(self, motion):
        """
        :param motion: [batch_size, num_frames, dim]
        """
        
        x = self.conv_layers(motion.permute(0, 2, 1))   # [batch_size, dim, npatches]
        validity = self.final(x).permute(0, 2, 1)       # [batch_size, npatches, 1]
        return validity

""" Conditional Discriminator.
1. Inputs are motion tokens sequence and condition embedding.
"""
class DiscriminatorV4(nn.Module):
    """
    ViT-like discriminator.
    """
    def __init__(self, n_tokens, cond_dim, channels, d_model, d_noise, 
                 num_heads, hidden_dim, num_layers, n_down, 
                 dropout, activation, **kwargs):
        super(DiscriminatorV4, self).__init__()
        assert len(channels) == n_down

        # Token embeddings
        self.token_emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=channels[0], padding_idx=None)
        
        # Convolution layers
        self.cond_linear = nn.Linear(cond_dim, d_model, bias=True)
        # Convolution layers
        layers = [
            nn.Conv1d(channels[0], channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.conv_layers = nn.Sequential(*layers)

        # Transformer layers
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                               nhead=num_heads, 
                                                               dim_feedforward=hidden_dim, 
                                                               dropout=dropout, 
                                                               activation=activation)
        self.tfm_layers = nn.TransformerEncoder(transformer_encoder_layer, 
                                                num_layers=num_layers)
        self.sequence_pose_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.noise_token = nn.Parameter(torch.zeros(d_model), requires_grad=True)

        self.final = nn.Linear(d_model, 1, bias=False)
        self.noise_final = nn.Linear(d_model, d_noise, bias=False)
    
    def forward(self, tokens, cond_emb):
        """
        :param motion: [batch_size, num_frames]
        :param cond_emb: [batch_size, 1, dim]
        """
        lengths = [x.shape[0] for x in tokens]
        batch_size = len(tokens)
        cond_x = self.cond_linear(cond_emb)
        cond_x = cond_x.permute(1, 0, 2)    # [1, batch_size, dim]

        x = self.token_emb(tokens)
        x = self.conv_layers(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = torch.cat((self.noise_token[None, None].repeat(batch_size, 1, 1), x), dim=1)
        xseq = self.sequence_pose_encoding(x)
        xseq = self.tfm_layers(xseq.permute(1, 0, 2) + cond_x).permute(1, 0, 2)
        
        noise = self.noise_final(xseq[:, :1])
        validity = self.final(xseq[:, 1:])
        
        return validity, noise



if __name__ == "__main__":
    D = DiscriminatorV4(n_tokens=2048, cond_dim=512, d_model=1024, 
                        d_noise=512, n_down=2, channels=[1024, 1024],
                        num_heads=8, hidden_dim=1024, num_layers=2, 
                        dropout=0.1, activation="gelu")
    tokens = torch.randint(0, 2048, size=(2, 64))
    cond_emb = torch.randn((2, 1, 512))
    validity, noise = D(tokens, cond_emb)
    print(validity.shape)
    print(noise.shape)
