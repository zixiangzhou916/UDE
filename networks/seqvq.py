import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
from .position_encoding import PositionalEncoding

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

""" Quantizers """
class Quantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, **kwargs):
        super(Quantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q - z.detach())**2) + self.beta * \
               torch.mean((z_q.detach() - z)**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))
        return loss, z_q, min_encoding_indices, perplexity

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        # B x 1
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def get_z_to_code_distance(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        z_flattened = z.contiguous().view(-1, self.e_dim)

        # B x V
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        return d.reshape(z.shape[0], z.shape[1], -1)

    def get_codebook_entry(self, indices):
        """

        :param indices(B, seq_len):
        :return z_q(B, seq_len, e_dim):
        """
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q

class FactorizedQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, use_proj=True, use_norm=True, use_regularize=False, **kwargs):
        super(FactorizedQuantizer, self).__init__()

        self.e_dim = e_dim
        self.n_e = n_e
        # self.f_dim = f_dim
        self.beta = beta
        self.use_proj = use_proj
        self.use_norm = use_norm
        self.use_reg = use_regularize

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # self.embedding.weight.data.normal_()
        nn.init.orthogonal_(self.embedding.weight)
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        # print('z', z.shape)
        assert z.shape[-1] == self.e_dim
        batch_size, num_codes, *_ = z.shape

        z_reshaped = z.contiguous().view(-1, self.e_dim)

        # Normalize the encoded latent code
        if self.use_norm:
            z_reshaped = F.normalize(z_reshaped, dim=-1)
            embedding_norm = F.normalize(self.embedding.weight, dim=-1)
        else:
            embedding_norm = self.embedding.weight

        d = torch.sum(z_reshaped ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.matmul(z_reshaped, embedding_norm.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        if self.use_norm:
            z_q_norm = F.normalize(z_q, dim=-1)
            z_norm = F.normalize(z, dim=-1)
            loss = torch.mean((z_q_norm - z_norm.detach())**2) + self.beta * \
                   torch.mean((z_q_norm.detach() - z_norm)**2)
        else:
            loss = torch.mean((z_q - z.detach())**2) + self.beta * \
                   torch.mean((z_q.detach() - z)**2)

        # compute the embedding regularization
        if self.use_reg:
            mask = torch.eye(num_codes).float().to(z.device)                     # [n_code, n_code]
            z_ = z_q.reshape(batch_size, num_codes, self.e_dim)
            emb_d = (z_[:, :, None].repeat(1, 1, num_codes, 1) - 
                     z_[:, None].repeat(1, num_codes, 1, 1)).norm(dim=-1)  # [bs, n_code, n_code]
            mean_d = emb_d.mean()
            reg_loss = (torch.exp(-(emb_d / 1.0)) * (1. - mask[None].repeat(batch_size, 1, 1))).mean()
            # print(' ---> reg_loss: ', reg_loss, "|", mean_d)
            loss = loss + 1.0 * reg_loss   # add embedding regularization loss

        # preserve gradients
        z_q = z + (z_q - z).detach()
        if self.use_norm:
            z_q = F.normalize(z_q, dim=-1)

        min_encodings = F.one_hot(min_encoding_indices, self.n_e).type(z.dtype)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean*torch.log(e_mean + 1e-10)))

        # loss = loss + 1.0 * torch.exp(-(perplexity / 100.))   # add perplexity regularization

        return loss, z_q, min_encoding_indices, perplexity

    def map2index(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vectort that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        :param z (B, seq_len, channel):
        :return z_q:
        """
        assert z.shape[-1] == self.e_dim
        batch_size, num_codes, *_ = z.shape

        z_reshaped = z.contiguous().view(-1, self.e_dim)

        # Normalize the encoded latent code
        if self.use_norm:
            z_reshaped = F.normalize(z_reshaped, dim=-1)
            embedding_norm = F.normalize(self.embedding.weight, dim=-1)
        else:
            embedding_norm = self.embedding.weight

        d = torch.sum(z_reshaped ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.matmul(z_reshaped, embedding_norm.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def soft_map2index(self, z):
        raise NotImplementedError

    def get_codebook_entry(self, indices):
        """
        Get codes from input indices.
        :param indices(B, seq_len):
        :return z_q(B, seq_len, e_dim):
        """
        index_flattened = indices.view(-1)
        z_q = self.embedding(index_flattened)
        if self.use_norm:
            z_q = F.normalize(z_q, dim=-1)
        z_q = z_q.view(indices.shape + (self.e_dim, )).contiguous()
        return z_q

""" VQEncoders """
class VQEncoder(nn.Module):
    def __init__(self, input_size, channels, n_down, **kwargs):
        super(VQEncoder, self).__init__()
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i-1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs  # [bs, nframes / 4, 1024]

""" VQDecoders """
class VQDecoder(nn.Module):
    """Conv + Transformer decoder.
    """
    def __init__(self, input_size, channels, n_resblk, n_up, hidden_dims, num_layers, num_heads, dropout, activation="gelu", **kwargs):
        super(VQDecoder, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        assert len(channels) == n_up + 1
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels[i], channels[i], kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        self.convs = nn.Sequential(*layers)
        self.convs.apply(init_weight)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=channels[-2],
                                                               nhead=num_heads,
                                                               dim_feedforward=hidden_dims,
                                                               dropout=dropout,
                                                               activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, 
                                                 num_layers=num_layers)
        self.transformer.apply(init_weight)
        self.sequence_pos_encoding = PositionalEncoding(channels[-2], dropout)

        self.linear = nn.Linear(channels[-2], channels[-1])

    def forward(self, inputs, noise=None):
        x = inputs.permute(0, 2, 1)                 # [batch_size, num_dims, num_frames]
        x = self.convs(x).permute(2, 0, 1)          # [num_frames, batch_size, num_dims]
        x = self.sequence_pos_encoding(x)
        x = self.transformer(x)
        outputs = self.linear(x).permute(1, 0, 2)   # [batcn_size, num_frames, num_dims]
        return outputs  # [bs, nframes, 263]

