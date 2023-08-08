import os
import sys
sys.path.append(os.getcwd())

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *
from .clip import clip
import numpy as np
import random
import math

def get_pad_mask(batch_size, seq_len, non_pad_lens):
    non_pad_lens = non_pad_lens.data.tolist()
    mask_2d = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    for i, cap_len in enumerate(non_pad_lens):
        mask_2d[i, :cap_len] = 1
    return mask_2d.unsqueeze(1).bool()

def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)

def get_clip_textencoder_mask(tokens):
    """
    :param tokens: [batch_size, 77]
    """
    mask = torch.zeros_like(tokens)
    for i in range(tokens.shape[0]):
        id = tokens[i].argmax(-1)
        mask[i, :id+1] = 1
    mask = mask.bool()
    return mask.unsqueeze(1)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def check_nan(tensor):
    has_nan = torch.isnan(tensor)
    if has_nan.float().sum() > 0: 
        print(tensor)

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
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
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

"""GPT, the noise is mapped by linear layers
1. VQGAN-GPT
"""
class UDETransformer(nn.Module):
    def __init__(self, conf):
        super(UDETransformer, self).__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.pad_idx = conf["decoder"]["pad_idx"]
        
        self.style_proj = nn.Linear(conf["decoder"]["d_model"], conf["decoder"]["d_latent"])
        
        layers = []
        for _ in range(conf["decoder"]["n_mlp"]):
            layers.append(
                nn.Linear(in_features=conf["decoder"]["d_latent"], out_features=conf["decoder"]["d_latent"])
            )
        self.style = nn.Sequential(*layers)
        
        # Encoder
        self.encoder = importlib.import_module(
            conf["encoder"]["arch_path"], package="networks").__getattribute__(
                conf["encoder"]["arch_name"])(**conf["encoder"])
        
        # Decoder
        self.decoder = importlib.import_module(
            conf["gpt"]["arch_path"], package="networks").__getattribute__(
                conf["gpt"]["arch_name"])(conf["gpt"])
            
        # Build CLIP
        self.clip_model, _ = clip.load(name=conf["clip"]["name"], 
                                       jit=conf["clip"]["jit"], 
                                       download_root=conf["clip"]["download_root"])
        for p in self.clip_model.parameters():
            p.requires_grad = False
        assert self.clip_model.training == False    # make sure CLIP is frozen
        
        # for n, p in self.clip_model.named_parameters():
        #     print(n, "|", p.dtype)
        # exit(0)        
        # print(self.encoder)
        # print(self.decoder)
    
    def train(self):
        self.style_proj.train()
        self.style.train()
        self.encoder.train()
        self.decoder.train()
        
    def eval(self):
        self.style_proj.eval()
        self.style.eval()
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, text, audio, trg_seq, src_non_pad_lens, tf_ratio, trg_start_index=8):
        raise NotImplementedError
    
    def text_to_motion(self, text, trg_seq, latent=None, tf_ratio=0.9):
        # Encode text input using CLIP
        with torch.no_grad():
            lengths = [len(s) for s in text]
            context_length = np.max(lengths)
            text_tokens = clip.tokenize(text, truncate=True).to(self.device)
            src_mask = get_clip_textencoder_mask(text_tokens)   # [batch_size, 1, num_frames]
            text_emb = self.clip_model.extract_text_embedding(text_tokens).float()  # [batch_size, num_frames, num_dims]
            # mask out the padding parts
            text_emb *= src_mask.permute(0, 2, 1).float()
        
        # Encode setence embedding and word embedding
        glob_emb, seq_emb = self.encoder.encode_text(t_seq=text_emb, 
                                                     m_seq=None, 
                                                     t_mask=src_mask[:, 0], 
                                                     m_mask=None)
        
        if latent is None:
            latent = self.style_proj(glob_emb)
        else:
            latent = self.style(latent) + self.style_proj(glob_emb)
            
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)
        
        # Decode using GPT architecture
        out = self.decoder(trg_seq, cond_emb)
        
        return out, glob_emb, seq_emb
    
    def audio_to_motion(self, audio, trg_seq, latent=None, trg_start_index=10, tf_ratio=0.9):
        # Encode the audio and motion primitives as cross-modality embedding
        a_mask = torch.ones(audio.shape[0], audio.shape[1]).bool().to(self.device)
        m_mask = torch.ones(audio.shape[0], trg_start_index).bool().to(self.device)
        glob_emb, seq_emb = self.encoder.encode_audio(a_seq=audio, 
                                                      m_seq=trg_seq[:, :trg_start_index], 
                                                      a_mask=a_mask, 
                                                      m_mask=m_mask)
        
        if latent is None:
            latent = self.style_proj(glob_emb)
        else:
            latent = self.style(latent) + self.style_proj(glob_emb)
            
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)
        
        # Decode using GPT architecture
        out = self.decoder(trg_seq[:, trg_start_index-1:], cond_emb)
        
        return out, glob_emb, seq_emb

    def sample_text_to_motion(self, text, trg_sos, trg_eos, latent, max_steps=80):
        
        trg_seq = torch.LongTensor(len(text), 1).fill_(trg_sos).long().to(self.device)
        batch_size = len(text)
        
        # Encode text input using CLIP
        with torch.no_grad():
            lengths = [len(s) for s in text]
            context_length = np.max(lengths)
            text_tokens = clip.tokenize(text, truncate=True).to(self.device)
            src_mask = get_clip_textencoder_mask(text_tokens)
            text_emb = self.clip_model.extract_text_embedding(text_tokens).float()
            # mask out the padding parts
            text_emb *= src_mask.permute(0, 2, 1).float()
            
        # Encode setence embedding and word embedding
        glob_emb, seq_emb = self.encoder.encode_text(t_seq=text_emb, 
                                                     m_seq=None, 
                                                     t_mask=src_mask[:, 0], 
                                                     m_mask=None)
        
        if latent is None:
            latent = self.style_proj(glob_emb)
        else:
            noise = self.style(latent.clone())
            latent = self.style(latent) + self.style_proj(glob_emb)
            
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)
        # print(latent[0, 0, :5], "|", noise[0, 0, :5], "|", cond_emb[0, 0, :5])
        
        # Decode using GPT architecture
        for _ in range(max_steps):
            out = self.decoder(trg_seq, cond_emb)
            probs = F.softmax(out[:, -1], dim=-1)
            # _, idx_out = torch.topk(probs, k=1, dim=-1)         # Deterministic
            idx_out = torch.multinomial(probs, num_samples=1)   # Random samples
            trg_seq = torch.cat((trg_seq, idx_out), dim=1)
        
        return trg_seq
    
    def sample_text_to_motion_v3(self, text, trg_sos, trg_eos, latent, max_steps=80):
        trg_seq = torch.LongTensor(len(text), 1).fill_(trg_sos).long().to(self.device)
        batch_size = len(text)
        
        # Encode text input using CLIP
        with torch.no_grad():
            lengths = [len(s) for s in text]
            context_length = np.max(lengths)
            text_tokens = clip.tokenize(text, truncate=True).to(self.device)
            src_mask = get_clip_textencoder_mask(text_tokens)
            text_emb = self.clip_model.extract_text_embedding(text_tokens).float()
            # mask out the padding parts
            text_emb *= src_mask.permute(0, 2, 1).float()
            
        # Encode setence embedding and word embedding
        glob_emb, seq_emb = self.encoder.encode_text(t_seq=text_emb, 
                                                     m_seq=None, 
                                                     t_mask=src_mask[:, 0], 
                                                     m_mask=None)
        
        if latent is None:
            latent = self.style_proj(glob_emb)
        else:
            noise = self.style(latent.clone())
            latent = self.style(latent) + self.style_proj(glob_emb)
            
        # And random noise to seq_emb
        seq_noise = torch.randn_like(seq_emb).normal_(std=0.5).float().to(seq_emb.device)
        # seq_noise = torch.clamp(seq_noise, min=-0.5, max=0.5)

        cond_emb = torch.cat((latent, seq_emb + seq_noise), dim=1)
        cond_length = cond_emb.size(1)

        # Decode using GPT architecture
        for _ in range(max_steps):
            out = self.decoder(trg_seq, cond_emb)
            probs = F.softmax(out[:, -1], dim=-1)
            _, idx_out = torch.topk(probs, k=1, dim=-1)
            trg_seq = torch.cat((trg_seq, idx_out), dim=1)
        
        return trg_seq

    def sample_text_to_motion_v4(self, text, trg_sos, trg_eos, latent, max_steps=80):
        # Stage one
        max_steps_stage_one = max_steps // 2
        pred_tokens = self.sample_text_to_motion(text, trg_sos, trg_eos, latent, max_steps=max_steps_stage_one)

        # Stage two
        # 1) Encode text input using CLIP
        with torch.no_grad():
            lengths = [len(s) for s in text]
            context_length = np.max(lengths)
            text_tokens = clip.tokenize(text, truncate=True).to(self.device)
            src_mask = get_clip_textencoder_mask(text_tokens)
            text_emb = self.clip_model.extract_text_embedding(text_tokens).float()
            # mask out the padding parts
            text_emb *= src_mask.permute(0, 2, 1).float()

        # 2) Start to predict tokens one-by-one
        mot_primitive_len = 8
        for _ in range(max_steps - max_steps_stage_one):
            # Encode setence embedding and word embedding
            m_mask = torch.ones(pred_tokens.shape[0], pred_tokens[:, -mot_primitive_len:].shape[1]).bool().to(self.device)
            glob_emb, seq_emb = self.encoder.encode_text(t_seq=text_emb, 
                                                         m_seq=pred_tokens[:, -mot_primitive_len:], 
                                                         t_mask=src_mask[:, 0], 
                                                         m_mask=m_mask)

            if latent is None:
                latent = self.style_proj(glob_emb)
            else:
                latent = self.style(latent) + self.style_proj(glob_emb)

            cond_emb = torch.cat((latent, seq_emb), dim=1)

            # Decode using GPT architecture
            out = self.decoder(pred_tokens[:, -1:], cond_emb)
            probs = F.softmax(out[:, -1], dim=-1)
            _, idx_out = torch.topk(probs, k=1, dim=-1)
            pred_tokens = torch.cat((pred_tokens, idx_out), dim=1)
            # print('input: ', pred_tokens[:, -1:], 'output: ', idx_out)

        return pred_tokens

    def sample_audio_to_motion(self, audio, trg_seq, trg_start, trg_sos, trg_eos, latent, max_steps=80):
        
        # Encode the audio and motion primitives as cross-modality embedding
        a_mask = torch.ones(audio.shape[0], audio.shape[1]).bool().to(self.device)
        m_mask = torch.ones(audio.shape[0], trg_seq.shape[1]).bool().to(self.device)
        glob_emb, seq_emb = self.encoder.encode_audio(a_seq=audio, 
                                                      m_seq=trg_seq, 
                                                      a_mask=a_mask, 
                                                      m_mask=m_mask)
        
        if latent is None:
            latent = self.style_proj(glob_emb)
        else:
            # noise = self.style(latent.clone())
            # print(noi)
            latent = self.style(latent) + self.style_proj(glob_emb)
            
        cond_emb = torch.cat((latent, seq_emb), dim=1)
        cond_length = cond_emb.size(1)
        
        # Decode using GPT architecture
        pred_seq = trg_start
        for _ in range(max_steps):
            out = self.decoder(pred_seq, cond_emb)
            probs = F.softmax(out[:, -1], dim=-1)
            # _, idx_out = torch.topk(probs, k=1, dim=-1)         # Deterministic
            idx_out = torch.multinomial(probs, num_samples=1)   # Random samples
            pred_seq = torch.cat((pred_seq, idx_out), dim=1)
            
        return pred_seq
    
    def sample_audio_to_motion_auto_regressive(self, audio, trg_seq, trg_start, trg_sos, trg_eos, latent, max_steps=80):
        
        audio_block_size = int(160 * 3)
        motion_block_size = 160 // 4
        
        # Stage one
        stage_one_max_steps = audio_block_size // 12 - trg_seq.size(1)
        total_pred_tokens = self.sample_audio_to_motion(audio[:, :audio_block_size], 
                                                        trg_seq=trg_seq, trg_start=trg_start, 
                                                        trg_sos=None, trg_eos=None, 
                                                        latent=latent, max_steps=stage_one_max_steps)
        total_pred_tokens = total_pred_tokens[:, 1:]
        total_pred_tokens = torch.cat((trg_seq, total_pred_tokens), dim=1)
        
        # Stage two
        mot_primitive_len = trg_seq.shape[1]
        a_idx = 1
        m_idx = 1
        for k in range(total_pred_tokens.size(1), max_steps, 1):
            audio_input = audio[:, a_idx*12:a_idx*12+audio_block_size]
            trg_seq_input = total_pred_tokens[:, m_idx:m_idx+mot_primitive_len]
            trg_start_input = total_pred_tokens[:, m_idx+mot_primitive_len-1:]
            pred_tokens = self.sample_audio_to_motion(audio=audio_input, 
                                                      trg_seq=trg_seq_input, 
                                                      trg_start=trg_start_input, 
                                                      trg_sos=None, trg_eos=None, 
                                                      latent=latent, max_steps=1)
            pred_tokens = pred_tokens[:, -1:]
            total_pred_tokens = torch.cat((total_pred_tokens, pred_tokens), dim=1)
            # print(' --- audio {:d} to {:d}, total_tokens {:d}'.format(a_idx*12, a_idx*12+ audio_block_size, total_pred_tokens.size(1)))
            
            a_idx += 1
            m_idx += 1
        
        # total_pred_tokens = torch.cat((trg_seq, total_pred_tokens), dim=1)
        
        return total_pred_tokens

    def encode_text_embedding(self, caption, input_token=False, requre_grad=False):
        if not requre_grad:
            with torch.no_grad():
                if not input_token:
                    text_tokens = clip.tokenize(caption, truncate=True).to(self.device)
                    text_emb = self.clip_model.encode_text(text_tokens).float()
                else:
                    if caption.dtype == torch.long:
                        text_emb = self.clip_model.encode_text(caption).float()
                    elif caption.dtype == torch.float:
                        text_emb = self.clip_model.encode_text_from_onehot(caption).float()
        else:
            if not input_token:
                text_tokens = clip.tokenize(caption, truncate=True).to(self.device)
                text_emb = self.clip_model.encode_text(text_tokens).float()
            else:
                if caption.dtype == torch.long:
                    text_emb = self.clip_model.encode_text(caption).float()
                elif caption.dtype == torch.float:
                    text_emb = self.clip_model.encode_text_from_onehot(caption.float()).float()
        
        return text_emb
    
    def get_clip_token_num(self):
        return self.clip_model.token_embedding.weight.shape[0]

if __name__ == '__main__':
    import yaml
    import numpy as np
    
    with open("configs/vqgan/config_vqgan_test_ude_gpt_v2.yaml", "r") as f:
        conf = yaml.safe_load(f)
        
    # amass = np.load("debug/vqgan/amass.npy", allow_pickle=True).item()
    # aist = np.load("debug/vqgan/aist.npy", allow_pickle=True).item()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data = np.load("debug_dict.npy", allow_pickle=True).item()

    audios = torch.from_numpy(data["audio"]).float().to(device)
    tokens = torch.from_numpy(data["token"]).long().to(device)
    token_sos = torch.from_numpy(data["token_sos"]).long().to(device)

    checkpoint = torch.load(conf["eval"]["ta2m_checkpoint"], map_location=torch.device("cpu"))
    
    model = importlib.import_module(
        conf["model"]["ude"]["arch_path"], package="networks").__getattribute__(
            conf["model"]["ude"]["arch_name"])(conf["model"]["ude"]).to(device)
    model.load_state_dict(checkpoint["ta2m_transformer"], strict=True)
    model.eval()
    # print(model)
    print("{:s} loaded successfully".format(conf["eval"]["ta2m_checkpoint"]))
        
    # model.text_to_motion(text=amass["caption"], 
    #                      trg_seq=torch.from_numpy(amass["motion"]).long().cuda(), 
    #                      latent=None)
    
    # model.audio_to_motion(audio=torch.from_numpy(aist["audio"]).float().cuda(), 
    #                       trg_seq=torch.from_numpy(aist["motion"]).long().cuda(), 
    #                       latent=None, 
    #                       trg_start_index=10)

    n_latent = conf["model"]["ude"]["decoder"]["d_latent"]
    # n_latent = 100
    all_tokens = []
    for _ in range(3):
        noise = torch.randn(1, 100, n_latent).normal_(std=1.0).float().to(device)
        noise = torch.clamp(noise, min=-1.0, max=1.0)

        # pred_tokens = model.sample_audio_to_motion_auto_regressive(
        #     audio=audios, trg_seq=tokens, trg_start=token_sos, 
        #     trg_sos=None, trg_eos=None, latent=noise, max_steps=80)

        # self.mot_start_idx = self.opt["model"]["quantizer"]["n_e"]
        # self.mot_end_idx = self.opt["model"]["quantizer"]["n_e"] + 1
        # self.mot_pad_idx = self.opt["model"]["quantizer"]["n_e"] + 2
        pred_tokens = model.sample_text_to_motion_v2(["man walks forward moving hands and neck."], 
                                                  trg_sos=conf["model"]["quantizer"]["n_e"], 
                                                  trg_eos=conf["model"]["quantizer"]["n_e"] + 1, 
                                                  latent=noise, max_steps=40)
        # pred_tokens = model.sample_text_to_motion(["a person stands, bent slightly forward, holds onto something with both hands"], 
        #                                           trg_sos=conf["model"]["quantizer"]["n_e"], 
        #                                           trg_eos=conf["model"]["quantizer"]["n_e"] + 1, 
        #                                           latent=noise, max_steps=60)
        all_tokens.append(pred_tokens[:, 1:])
    
    for t in all_tokens: 
        print(t[0].detach().cpu().numpy())
    
    # tokens = torch.cat(tokens, dim=1)
    # min_tokens = tokens.min(dim=0)[0]
    # max_tokens = tokens.max(dim=0)[0]
    # print(min_tokens)
    # print(max_tokens)
    
    
    
    