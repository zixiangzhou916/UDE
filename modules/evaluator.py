import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from os.path import join as pjoin
from datetime import datetime
import importlib
from tqdm import tqdm
import json

from funcs.logger import setup_logger
from funcs.comm_utils import get_rank

from smplx import SMPL

class UDEEvaluator(object):
    def __init__(self, args, opt):
        self.args = args
        self.opt = opt
        self.animation_dir = os.path.join(self.args.eval_folder, self.args.eval_name, "animation")
        self.output_dir = os.path.join(self.args.eval_folder, self.args.eval_name, "output")
        os.makedirs(os.path.join(self.args.eval_folder, self.args.eval_name), exist_ok=True)
        self.logger = setup_logger('UDE', os.path.join(self.args.eval_folder, self.args.eval_name), get_rank(), filename='evaluation_log.txt')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.setup_models()
        # self.setup_loaders()
        
    def setup_models(self):
        self.mot_start_idx = self.opt["model"]["quantizer"]["n_e"]
        self.mot_end_idx = self.opt["model"]["quantizer"]["n_e"] + 1
        self.mot_pad_idx = self.opt["model"]["quantizer"]["n_e"] + 2
        
        self.sos = torch.tensor(self.mot_start_idx).to(self.device)
        self.eos = torch.tensor(self.mot_end_idx).to(self.device)
        self.pos = torch.tensor(self.mot_pad_idx).to(self.device)
        
        try:
            vq_checkpoint = torch.load(self.opt["eval"]["vq_checkpoint"], map_location=torch.device("cpu"))

            self.vq_encoder = importlib.import_module(
                f".seqvq", package="networks").__getattribute__(
                    self.opt["model"]["vq_encoder"]["arch"])(**self.opt["model"]["vq_encoder"])
            self.vq_encoder.load_state_dict(vq_checkpoint["vq_encoder"], strict=True)
            self.vq_encoder = self.vq_encoder.to(self.device)
            self.vq_encoder.eval()
            self.logger.info("VQEncoder loaded from {:s} successfully".format(self.opt["eval"]["vq_checkpoint"]))
            
            self.quantizer = importlib.import_module(
                f".seqvq", package="networks").__getattribute__(
                    self.opt["model"]["quantizer"]["arch"])(**self.opt["model"]["quantizer"])
            self.quantizer.load_state_dict(vq_checkpoint["quantizer"], strict=True)
            self.quantizer = self.quantizer.to(self.device)
            self.quantizer.eval()
            self.logger.info("Quantizer loaded from {:s} successfully".format(self.opt["eval"]["vq_checkpoint"]))

            if not self.args.use_dmd:
                self.vq_decoder = importlib.import_module(
                    f".seqvq", package="networks").__getattribute__(
                        self.opt["model"]["vq_decoder"]["arch"])(**self.opt["model"]["vq_decoder"])
                self.vq_decoder.load_state_dict(vq_checkpoint["vq_decoder"], strict=True)
                self.vq_decoder = self.vq_decoder.to(self.device)
                self.vq_decoder.eval()
                self.logger.info("VQDecoder loaded from {:s} successfully".format(self.opt["eval"]["vq_checkpoint"]))
        except:
            raise RuntimeError("VQ checkpoints loaded failed!!!")
        
        try:
            ta2m_checkpoint = torch.load(self.opt["eval"]["ta2m_checkpoint"], map_location=torch.device("cpu"))
            self.ta2m_model = importlib.import_module(
                self.opt["model"]["ude"]["arch_path"], package="networks").__getattribute__(
                    self.opt["model"]["ude"]["arch_name"])(self.opt["model"]["ude"])
            self.ta2m_model.load_state_dict(ta2m_checkpoint["ta2m_transformer"], strict=True)
            self.ta2m_model = self.ta2m_model.to(self.device)
            self.ta2m_model.eval()
            self.logger.info("UDE loaded from {:s} successfully".format(self.opt["eval"]["ta2m_checkpoint"]))
        except:
            raise RuntimeError("UDE loaded failed!!!")
        
        if self.args.use_dmd:
            try:
                dmd_checkpoint = torch.load(self.opt["eval"]["diffusion_checkpoint"], map_location=torch.device("cpu"))
                self.dmd_model = importlib.import_module(
                    self.opt["model"]["dmd"]["arch_path"], package="networks").__getattribute__(
                        self.opt["model"]["dmd"]["arch_name"])(self.opt["model"]["dmd"])
                self.dmd_model.encoder.load_state_dict(dmd_checkpoint["diffusion"], strict=True)
                self.dmd_model.eval()
                self.logger.info("DMD loaded from {:s} successfully".format(self.opt["eval"]["diffusion_checkpoint"]))
            except:
                raise RuntimeError("DMD loaded failed!!!")
        
    def setup_loaders(self):
        self.eval_loader, self.eval_dataset = importlib.import_module(
            ".{:s}".format(self.opt["data"]["dataset"]["module"]), package="dataset").__getattribute__(
                f"get_eval_ta2m_dataloader")(self.opt)
            
    def sample_noise(self, batch_size, length):
        if self.args.dist_type == "uniform":
            scale = self.args.noise_scale
            n_latent = self.opt["model"]["ude"]["decoder"]["d_latent"]
            noise = torch.randn(batch_size, length, n_latent).uniform_(-scale, scale).float().to(self.device)
            noise = torch.clamp(noise, min=-1.0, max=1.0)
        elif self.args.dist_type == "normal":
            scale = self.args.noise_scale
            n_latent = self.opt["model"]["ude"]["decoder"]["d_latent"]
            noise = torch.randn(batch_size, 1, n_latent).normal_(std=scale).float().to(self.device)
            noise = torch.clamp(noise, min=-1.0, max=1.0)
        else:
            noise = None
        return noise
    
    def eval_text_to_motion(self, captions, targ_len, batch_id):
        captions = [s for s in captions]
        batch_size = len(captions)
        
        for t in range(self.args.repeat_times):
            
            if self.args.add_noise:
                noise_length = 1
                noise = self.sample_noise(batch_size=batch_size, length=noise_length)
            else:
                noise = None
            
            with torch.no_grad():
                pred_tokens = self.ta2m_model.sample_text_to_motion(text=captions, 
                                                                    trg_sos=self.mot_start_idx, 
                                                                    trg_eos=self.mot_end_idx, 
                                                                    latent=noise, 
                                                                    max_steps=targ_len // 4)

                if not self.args.use_dmd:
                    # Use VQDecoder
                    pred_tokens = pred_tokens[:, 1:]
                    vq_latent = self.quantizer.get_codebook_entry(pred_tokens)
                    gen_motion = self.vq_decoder(vq_latent)
                    # print(pred_tokens[0].detach().cpu().numpy())   # DEBUG
                else:
                    # Use DiffusionMotionDecoder
                    motion_shape = (self.args.repeat_times_dmd, targ_len, 75)
                    pred_tokens = pred_tokens[:, 1:]
                    gen_motion = self.dmd_model.sample(
                        batch_data={
                            "motion": torch.zeros(motion_shape).float().to(self.device), 
                            "tokens": pred_tokens
                        }
                    )
            
            for t_ in range(gen_motion.size(0)):
                tid = t_ + t * self.args.repeat_times_dmd
                output_dict = {"motion": gen_motion[t_:t_+1].detach().cpu().numpy(),    # [bs, dim, nframes]
                               "caption": captions}

                output_path = os.path.join(self.output_dir, "t2m")
                os.makedirs(output_path, exist_ok=True)
                np.save(os.path.join(output_path, "B{:04d}_T{:03d}.npy".format(batch_id, tid)), output_dict)
            
    def eval_audio_to_motion(self, audio_name, audios, motions, batch_id):
        audios = audios.detach().to(self.device).float()
        motions = motions.detach().to(self.device).float()
        audio_name = [str(s) for s in audio_name]
        batch_size = len(audios)
        
        # Encode motion tokens
        with torch.no_grad():
            motion_emb = self.vq_encoder(motions)
            motion_token = self.quantizer.map2index(motion_emb)
            motion_token = motion_token.reshape(motion_emb.shape[:2])
            motion_token = torch.cat((self.sos[None].repeat(batch_size, 1), 
                                      motion_token), dim=1)  # <SOS>, <TOKEN_0>, <TOKEN_1>, ...
        
        tid = 0
        for t in range(self.args.repeat_times):
            if self.args.add_noise:
                noise_length = 1
                noise = self.sample_noise(batch_size=batch_size, length=noise_length)
            else:
                noise = None
            
            primitive_length = 8
            max_steps = audios.shape[1] // 12 - primitive_length + 1
                    
            with torch.no_grad():
                pred_tokens = self.ta2m_model.sample_audio_to_motion_auto_regressive(
                    audio=audios, trg_seq=motion_token[:, :primitive_length], 
                    trg_start=motion_token[:, primitive_length-1:primitive_length], 
                    trg_sos=None, trg_eos=None, latent=noise, max_steps=max_steps)
                
                pred_tokens = pred_tokens[:, 1:]

                for i in range(0, pred_tokens.shape[1] - self.args.audio_seg_len, self.args.audio_seg_len):
                    
                    if not self.args.use_dmd:
                        # Use VQDecoder
                        vq_latent = self.quantizer.get_codebook_entry(pred_tokens[:, i:i+self.args.audio_seg_len])
                        gen_motion = self.vq_decoder(vq_latent)
                    else:
                        # Use DiffusionMotionDecoder
                        motion_shape = (self.args.repeat_times_dmd, int(4 * self.args.audio_seg_len), motions.size(2))
                        gen_motion = self.dmd_model.sample(
                            batch_data={
                                "motion": torch.zeros(motion_shape).float().to(self.device), 
                                "tokens": pred_tokens[:, i:i+self.args.audio_seg_len]
                            }
                        )

                    for t_ in range(gen_motion.size(0)):
                        # print(audios.shape, "|", motions.shape, "|", gen_motion.shape)            
                        output_dict = {"motion": gen_motion[t_:t_+1].detach().cpu().numpy(),    # [bs, dim, nframes]
                                       "audio": audios[:, i*12:(i+self.args.audio_seg_len)*12].permute(0, 2, 1).cpu().numpy(), 
                                       "caption": audio_name}
    
                        output_path = os.path.join(self.output_dir, "a2m")
                        os.makedirs(output_path, exist_ok=True)
                        np.save(os.path.join(output_path, "B{:04d}_T{:03d}.npy".format(batch_id, tid)), output_dict)
                        tid += 1

    def generate_t2m(self, condition_input):
        import json
        
        self.ta2m_model.eval()
        self.vq_encoder.eval()
        self.quantizer.eval()
        if self.args.use_dmd:
            self.dmd_model.eval()
        else:
            self.vq_decoder.eval()
            
        with open(condition_input, "r") as f:
            text_descriptions = json.load(f)
        
        for batch_id, batch in tqdm(enumerate(text_descriptions)):
            text = batch.get("caption")
            self.eval_text_to_motion([text], targ_len=160, batch_id=batch_id)
    
    def generate_a2m(self, condition_input):
        
        self.ta2m_model.eval()
        self.vq_encoder.eval()
        self.quantizer.eval()
        if self.args.use_dmd:
            self.dmd_model.eval()
        else:
            self.vq_decoder.eval()
            
        files = [f for f in os.listdir(condition_input) if ".npy" in f]
        for batch_id, batch in tqdm(enumerate(files)):
            audio_name = batch.replace(".npy", "")
            data = np.load(os.path.join(condition_input, batch), allow_pickle=True).item()
            audio = torch.from_numpy(data.get("audio_sequence")).float().to(self.device)
            motion_primitives = torch.from_numpy(data.get("motion_smpl")).float().to(self.device)
            self.eval_audio_to_motion(audio_name, audio.unsqueeze(dim=0), motion_primitives.unsqueeze(dim=0), batch_id)
            
            
        