from cmath import exp
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from os.path import join as pjoin
import random
import json
import codecs as cs
from tqdm import tqdm
from scipy import ndimage

from torch.utils.data._utils.collate import default_collate
from .word_vectorizer import WordVectorizerV2

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def normalize_trans(motion):
    """
    :param motion: [num_frames, num_dims]
    """
    glob_trans = motion[:, :3]  # [num_frames, 3]
    avg_trans = np.mean(glob_trans, axis=0, keepdims=True)  # [1, 3]
    motion[:, :3] -= avg_trans
    return motion

""" VQVAE dataloader """
class MotionTokenizerDataset(data.Dataset):
    def __init__(self, opt, amass_split_file, aist_split_file, meta_dir):
        super(MotionTokenizerDataset, self).__init__()
        self.opt = opt

        amass_id_list = []
        aist_id_list = []
        with cs.open(amass_split_file, "r") as f:
            for line in f.readlines():
                amass_id_list.append(line.strip())
        
        with cs.open(aist_split_file, "r") as f:
            for line in f.readlines():
                aist_id_list.append(line.strip()) 

        self.data = []
        self.lengths = []
        for name in tqdm(amass_id_list):
            try:
                motion = np.load(pjoin(opt["amass_motion_dir"], name + '.npy'))
                if motion.shape[0] < opt["window_size"]:
                    continue
                self.lengths.append(motion.shape[0] - opt["window_size"])
                self.data.append(motion)
            except:
                pass

        for name in tqdm(aist_id_list):
            try:
                data = np.load(pjoin(opt["aist_motion_dir"], name + '.npy'), allow_pickle=True).item()
                motion = data["motion_smpl"]

                # downsample, the target fps is 20, and the original fps is 60
                motion = motion[::3]
                if motion.shape[0] < opt["window_size"]:
                    continue
                self.lengths.append(motion.shape[0] - opt["window_size"])
                self.data.append(motion)

            except:
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0

        motion = self.data[motion_id][idx:idx+self.opt["window_size"]]
        return motion

class TokenizerDataset(data.Dataset):
    def __init__(self, opt, amass_split_file, aist_split_file, meta_dir):
        super(TokenizerDataset, self).__init__()
        self.opt = opt
        min_motion_len = 40 if self.opt["dataset_name"] =='t2m' else 24

        amass_id_list = []
        aist_id_list = []
        for split_file in amass_split_file:
            with cs.open(split_file, "r") as f:
                for line in f.readlines():
                    amass_id_list.append(line.strip())
        
        for split_file in aist_split_file:
            with cs.open(split_file, "r") as f:
                for line in f.readlines():
                    aist_id_list.append(line.strip()) 

        data_dict = {}
        new_name_list = []
        length_list = []

        for name in tqdm(amass_id_list):
            try:
                motion = np.load(pjoin(opt["amass_motion_dir"], name + '.npy'))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                data_dict[name] = {"motion": motion, "length": len(motion), "name": name}
                new_name_list.append(name)
                length_list.append(len(motion))

            except:
                pass

        for name in tqdm(aist_id_list):
            try:
                data = np.load(pjoin(opt["aist_motion_dir"], name + '.npy'), allow_pickle=True).item()
                motion = data["motion_smpl"]
                # downsample, the target fps is 20, and the original fps is 60
                motion = motion[::3]
                if (len(motion)) < min_motion_len:
                    continue
                data_dict[name] = {"motion": motion, "length": len(motion), "name": name}
                new_name_list.append(name)
                length_list.append(len(motion))
                
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data["motion"], data["length"]
        m_length = (m_length // self.opt["unit_length"]) * self.opt["unit_length"]

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        return motion, name

""" Motion VAE dataloader """
class MotionVAEDataset(data.Dataset):
    def __init__(self, opt, amass_split_file, aist_split_file):
        super(MotionVAEDataset, self).__init__()
        self.opt = opt
        
        amass_id_list = []
        aist_id_list = []
        with cs.open(amass_split_file, "r") as f:
            for line in f.readlines():
                amass_id_list.append(line.strip())
        
        with cs.open(aist_split_file, "r") as f:
            for line in f.readlines():
                aist_id_list.append(line.strip())
        
        self.data_list = []
        self.data_list += self.read_all_amass(amass_id_list)
        self.data_list += self.read_all_aist(aist_id_list)
        
    def read_all_amass(self, amass_id_list):
        data_list = []
        for name in tqdm(amass_id_list):
            try:
                # Read motion
                motion = np.load(pjoin(self.opt["amass_motion_dir"], name + ".npy"))
                data_list.append(motion)
            except:
                pass
        return data_list
    
    def read_all_aist(self, aist_id_list):
        data_list = []
        for name in tqdm(aist_id_list):
            try:
                # Read audio
                pickle_data = np.load(pjoin(self.opt["aist_motion_dir"], name + ".npy"), allow_pickle=True).item()
                # Read motion
                motion = pickle_data["motion_smpl"][::3]
                data_list.append(motion)
            except:
                pass
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        motion = self.data_list[index]
        mot_len = motion.shape[0]
        if mot_len > self.opt["window_size"]:
            m_start_idx = np.random.randint(0, mot_len - self.opt["window_size"])
            m_end_idx = m_start_idx + self.opt["window_size"]
            motion = motion[m_start_idx:m_end_idx]
        else:
            motion = ndimage.zoom(motion, (self.opt["window_size"] / mot_len, 1), order=1)
        
        return motion

""" Motion Diffusion Auto-Encoder dataloader """
class MotionDiffusionAEDataset(data.Dataset):
    def __init__(self, opt, amass_split_file, aist_split_file, meta_dir=None):
        super(MotionDiffusionAEDataset, self).__init__()
        self.opt = opt

        amass_id_list = []
        aist_id_list = []
        with cs.open(amass_split_file, "r") as f:
            for line in f.readlines():
                amass_id_list.append(line.strip())
        
        with cs.open(aist_split_file, "r") as f:
            for line in f.readlines():
                aist_id_list.append(line.strip())
                
        self.data = []
        self.lengths = []
        for name in tqdm(amass_id_list):
            try:
                motion = np.load(pjoin(opt["amass_motion_dir"], name + '.npy'))
                if motion.shape[0] < opt["window_size"]:
                    motion = ndimage.zoom(motion, (self.opt["window_size"] / motion.shape[0], 1), order=1)
                self.lengths.append(motion.shape[0] - opt["window_size"])
                self.data.append(motion)
            
            except:
                pass
        
        for name in tqdm(aist_id_list):
            try:
                data = np.load(pjoin(opt["aist_motion_dir"], name + '.npy'), allow_pickle=True).item()
                motion = data["motion_smpl"]
                # Downsample to fps = 20
                motion = motion[::3]
                if motion.shape[0] < opt["window_size"]:
                    motion = ndimage.zoom(motion, (self.opt["window_size"] / motion.shape[0], 1), order=1)
                self.lengths.append(motion.shape[0] - opt["window_size"])
                self.data.append(motion)
            except:
                pass
            
        self.cumsum = np.cumsum([0] + self.lengths)
        
    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0

        motion = self.data[motion_id][idx:idx+self.opt["window_size"]]
        
        return motion

""" Read motions in fixed lengths """
class TextAudio2MotionTokenDataset(data.Dataset):
    def __init__(self, opt, amass_split_file, aist_split_file):
        super(TextAudio2MotionTokenDataset, self).__init__()
        self.opt = opt

        amass_id_list = []
        aist_id_list = []
        with cs.open(amass_split_file, "r") as f:
            for line in f.readlines():
                amass_id_list.append(line.strip())
        
        with cs.open(aist_split_file, "r") as f:
            for line in f.readlines():
                aist_id_list.append(line.strip())
                        
        amass_data_dict, amass_name_list = self.read_all_amass(amass_id_list)
        aist_data_dict, aist_name_list = self.read_all_aist(aist_id_list)
        
        if len(amass_name_list) < len(aist_name_list):
            idx = 0
            while len(amass_name_list) < len(aist_name_list):
                amass_name_list.append(amass_name_list[idx])
                idx += 1
        elif len(amass_name_list) > len(aist_name_list):
            idx = 0
            while len(aist_name_list) < len(amass_name_list):
                aist_name_list.append(aist_name_list[idx])
                idx += 1

        self.data_dict = {}
        self.data_dict.update(amass_data_dict)
        self.data_dict.update(aist_data_dict)

        self.name_list = [{"amass": amass, "aist": aist} for (amass, aist) in zip(amass_name_list, aist_name_list)]
        
    def read_all_amass(self, amass_id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(amass_id_list):
            try:
                # Read motion
                motion = np.load(pjoin(self.opt["amass_motion_dir"], name + ".npy"))
                # if motion.shape[0] < self.opt["amass_window_size"]: continue
                # m_start_id = np.random.randint(0, max(1, motion.shape[0] - self.opt["amass_window_size"]))
                # m_end_id = m_start_id + self.opt["amass_window_size"]
                # motion = motion[m_start_id:m_end_id]
                # Read text
                with cs.open(pjoin(self.opt["text_dir"], name + ".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)
                                data_dict[new_name] = {'text':[text_dict], "motion": motion}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'text': text_data, "motion": motion}
                    new_name_list.append(name)
            except:
                pass
            
        return data_dict, new_name_list
    
    def read_all_aist(self, aist_id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(aist_id_list):
            try:
                # Read audio
                pickle_data = np.load(pjoin(self.opt["aist_motion_dir"], name + ".npy"), allow_pickle=True).item()
                audio = pickle_data["audio_sequence"]
                
                # Read motion
                motion = pickle_data["motion_smpl"]
                # motion = motion[::3]
                if motion.shape[0] < 3 * self.opt["aist_window_size"] or audio.shape[0] < 3 * self.opt["aist_window_size"]: 
                    continue
                # m_start_id = np.random.choice(0, max(1, motion.shape[1] - self.opt["aist_window_size"]))
                # m_end_id = m_start_id + self.opt["aist_window_size"]
                # motion = motion[m_]
                
                data_dict[name] = {'audio': audio, 'motion': motion}
                
                new_name_list.append(name)
                
            except:
                pass
            
        return data_dict, new_name_list
    
    def __len__(self):
        return len(self.name_list) * self.opt["times"]
    
    def __getitem__(self, index):
        item = index % len(self.name_list)
        # print(' --- ', self.name_list[item])
        amass_name = self.name_list[item]["amass"]
        aist_name = self.name_list[item]["aist"]

        amass_data = self.data_dict[amass_name]
        aist_data = self.data_dict[aist_name]
        
        amass_output = self.get_one_amass(amass_data)
        aist_output = self.get_one_aist(aist_data)
        
        return {"amass": amass_output, "aist": aist_output}
        
    def get_one_amass(self, data):
        motion, text_list = data["motion"], data["text"]
        
        text_data = random.choice(text_list)
        caption, t_tokens = text_data['caption'], text_data['tokens']
        
        mot_len = motion.shape[0]
        
        if mot_len > self.opt["amass_window_size"]:
            m_start_idx = np.random.randint(0, mot_len - self.opt["amass_window_size"])
            m_end_idx = m_start_idx + self.opt["amass_window_size"]
            motion = motion[m_start_idx:m_end_idx]
        else:
            motion = ndimage.zoom(motion, (self.opt["amass_window_size"] / mot_len, 1), order=1)
        
        return caption, motion
    
    def get_one_aist(self, data):
        motion, audio = data["motion"], data["audio"]

        mot_len = motion.shape[0] 
        if mot_len > 3 * self.opt["aist_window_size"]:
            m_start_idx = np.random.randint(0, mot_len - 3 * self.opt["aist_window_size"])
            m_end_idx = m_start_idx + int(3 * self.opt["aist_window_size"])
            motion = motion[m_start_idx:m_end_idx]
            audio = audio[m_start_idx:m_end_idx]
        else:
            motion = ndimage.zoom(motion, (3 * self.opt["aist_window_size"] / mot_len, 1), order=1)
        
        # Downsample to 20fps
        motion = motion[::3]

        return audio, motion

""" Read motions as their tokens (different lengths). """
class TextAudio2MotionTokenDataset_v2(data.Dataset):
    def __init__(self, opt, amass_split_file, aist_split_file):
        super(TextAudio2MotionTokenDataset_v2, self).__init__()
        self.opt = opt

        amass_id_list = []
        aist_id_list = []
        with cs.open(amass_split_file, "r") as f:
            for line in f.readlines():
                amass_id_list.append(line.strip())
        
        with cs.open(aist_split_file, "r") as f:
            for line in f.readlines():
                aist_id_list.append(line.strip())

        amass_data_dict, amass_name_list = self.read_all_amass(amass_id_list)
        aist_data_dict, aist_name_list = self.read_all_aist(aist_id_list)

        if len(amass_name_list) < len(aist_name_list):
            idx = 0
            while len(amass_name_list) < len(aist_name_list):
                amass_name_list.append(amass_name_list[idx])
                idx += 1
        elif len(amass_name_list) > len(aist_name_list):
            idx = 0
            while len(aist_name_list) < len(amass_name_list):
                aist_name_list.append(aist_name_list[idx])
                idx += 1

        self.data_dict = {}
        self.data_dict.update(amass_data_dict)
        self.data_dict.update(aist_data_dict)

        self.name_list = [{"amass": amass, "aist": aist} for (amass, aist) in zip(amass_name_list, aist_name_list)]
        # print(self.name_list)
        # print(self.modality_list)
        # exit(0)
    
    def read_all_amass(self, amass_id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(amass_id_list):
            try:
                m_token_list = []
                # Read tokens
                with cs.open(pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name)), "r") as f:
                    for line in f.readlines():
                        m_token_list.append(line.strip().split(' '))
                
                # has_empty = False
                # for m_ in m_token_list:
                #     if len(m_) == 0: 
                #         print(' --- empty')
                #         has_empty = True
                
                # if has_empty:
                #     continue
                # print('read {:s}'.format(pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name))))

                # Read motion
                if self.opt["load_motion"]:
                    motion = np.load(pjoin(self.opt["amass_motion_dir"], name + ".npy"))

                # Read text
                with cs.open(pjoin(self.opt["text_dir"], name + ".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list = [tokens[int(f_tag*5) : int(to_tag*5)] for tokens in m_token_list]
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                has_empty = False
                                for m_ in m_token_list:
                                    if len(m_) == 0: 
                                        print(' --- 1 empty')
                                        has_empty = True
                                
                                if has_empty:
                                    continue
                                
                                if self.opt["load_motion"]:
                                    data_dict[new_name] = {'m_token_list': m_token_list,
                                                           'text':[text_dict], 
                                                           "motion": motion}
                                else:
                                    data_dict[new_name] = {'m_token_list': m_token_list,
                                                           'text':[text_dict]}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    has_empty = False
                    for m_ in m_token_list:
                        if len(m_) == 0: 
                            print(' --- 2 empty')
                            has_empty = True
                    
                    if has_empty:
                        continue

                    if self.opt["load_motion"]:
                        data_dict[name] = {'m_token_list': m_token_list, 
                                           'text': text_data, 
                                           "motion": motion}
                    else:
                         data_dict[name] = {'m_token_list': m_token_list, 
                                           'text': text_data}
                    new_name_list.append(name)
            except:
                pass

        return data_dict, new_name_list

    def read_all_aist(self, aist_id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(aist_id_list):
            
            try:
                m_token_list = []
                # Read tokens
                with cs.open(pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name)), "r") as f:
                    for line in f.readlines():
                        m_token_list.append(line.strip().split(' '))
                # print(pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name)))

                has_empty = False
                for m_ in m_token_list:
                    if len(m_) == 0: 
                        print(' --- aist empty')
                        has_empty = True
                
                if has_empty:
                    continue

                # Read audio
                pickle_data = np.load(pjoin(self.opt["aist_motion_dir"], name + ".npy"), allow_pickle=True).item()
                audio = pickle_data["audio_sequence"]
                
                # Read motion
                if self.opt["load_motion"]:
                    motion = pickle_data["motion_smpl"]

                if self.opt["load_motion"]:
                    data_dict[name] = {'m_token_list': m_token_list, 
                                       'audio': audio, 
                                       'motion': motion}
                else:
                    data_dict[name] = {'m_token_list': m_token_list, 
                                       'audio': audio}
                
                new_name_list.append(name)

            except:
                pass

        return data_dict, new_name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        # print(' --- ', self.name_list[item])
        amass_name = self.name_list[item]["amass"]
        aist_name = self.name_list[item]["aist"]

        amass_data = self.data_dict[amass_name]
        aist_data = self.data_dict[aist_name]
        
        amass_output = self.get_one_amass(amass_data)

        aist_output = self.get_one_aist(aist_data)
        
        if self.opt["load_motion"]:
            amass_caption, amass_m_tokens, amass_cap_lens, amass_m_tokens_len, amass_motion = amass_output
            aist_audio, aist_m_tokens, aist_aud_len, aist_m_tokens_len, aist_motion = aist_output

            # print(amass_motion.shape, "|", aist_motion.shape)
            print(amass_m_tokens.shape, "|", aist_m_tokens.shape)
        else:
            amass_caption, amass_m_tokens, amass_cap_lens, amass_m_tokens_len = amass_output
            aist_audio, aist_m_tokens, aist_aud_len, aist_m_tokens_len = aist_output

        return {"amass": amass_output, "aist": aist_output}

    def get_one_amass(self, data):
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)
        m_tokens = [int(token) for token in m_tokens]

        if len(m_tokens) == 0:
            print('=' * 10)
            for m in m_token_list: print(m)
            exit(0)

        text_data = random.choice(text_list)
        caption, t_tokens = text_data['caption'], text_data['tokens']
        if self.opt["load_motion"]:
            motion = data["motion"]

        if len(t_tokens) < self.opt["max_text_len"]:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt["max_text_len"] + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt["max_text_len"]]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
        
        m_tokens_len = len(m_tokens)
        m_tokens = [self.opt["mot_start_idx"]] + \
                   m_tokens + \
                   [self.opt["mot_end_idx"]] + \
                   [self.opt["mot_pad_idx"]] * (self.opt["max_amass_motion_len"] - len(m_tokens) - 2)
        
        m_tokens = np.array(m_tokens, dtype=int)

        if self.opt["load_motion"]:
            return caption, m_tokens, sent_len, m_tokens_len, motion
        else:
            return caption, m_tokens, sent_len, m_tokens_len

    def get_one_aist(self, data):
        m_token_list = data['m_token_list']
        m_tokens = random.choice(m_token_list)
        m_tokens_len = len(m_tokens)
        aud_len = int(3 * self.opt["unit_length"] * m_tokens_len)
        max_aud_len = int(3 * (self.opt["max_aist_motion_len"] - 2) * self.opt["unit_length"])

        if len(m_tokens) == 0:
            print('=' * 10)
            for m in m_token_list: print(m)
            exit(0)
        
        # Get audio of length associated with motion tokens
        audio = data["audio"][:int(3 * self.opt["unit_length"] * m_tokens_len)]
        aud_len = audio.shape[0]

        # Crop motion tokens and audio for data augmentation
        m_start_id = np.random.randint(0, m_tokens_len - self.opt["aist_window_size"])
        m_end_id = m_start_id + self.opt["aist_window_size"]
        a_start_id = m_start_id * 3 * self.opt["unit_length"]
        a_end_id = a_start_id + 3 * self.opt["unit_length"] * self.opt["aist_window_size"]

        # print(m_start_id, m_end_id, a_start_id, a_end_id, audio.shape)

        m_tokens = [self.opt["mot_start_idx"]] + \
                    m_tokens[m_start_id:m_end_id] + \
                   [self.opt["mot_end_idx"]] + \
                   [self.opt["mot_pad_idx"]] * (self.opt["max_aist_motion_len"] - self.opt["aist_window_size"] - 2)
        m_tokens = np.array(m_tokens, dtype=int)
        audio = audio[a_start_id:a_end_id]

        if self.opt["load_motion"]:
            motion = data["motion"][:, :aud_len]
            m_start_id_ = m_start_id * self.opt["unit_length"]
            m_end_id_ = m_start_id_ + self.opt["aist_window_size"] * self.opt["unit_length"]
            motion = motion[::3]
            motion = motion[m_start_id_:m_end_id_]

        if self.opt["load_motion"]:
            return audio, m_tokens, audio.shape[0], m_tokens.shape[0], motion
        else:
            return audio, m_tokens, audio.shape[0], m_tokens.shape[0]

""" Read motions in fixed lengths. 
Also read word tokens for motion-to-text cycle loss.
"""
class TextAudio2MotionTokenDataset_cycle(TextAudio2MotionTokenDataset):
    def __init__(self, opt, amass_split_file, aist_split_file, w_vectorizer):
        super(TextAudio2MotionTokenDataset_cycle, self).__init__(opt, amass_split_file, aist_split_file)
        self.w_vectorizer = w_vectorizer
        
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def get_one_amass(self, data):
        motion, text_list = data["motion"], data["text"]
        
        text_data = random.choice(text_list)
        caption, t_tokens = text_data['caption'], text_data['tokens']
        
        if len(t_tokens) < self.opt["max_text_len"]:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt["max_text_len"] + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt["max_text_len"]]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
        word_embeddings = []
        word_ids = []
        for i, t_token in enumerate(t_tokens):
            word_emb, _, word_id = self.w_vectorizer[t_token]
            word_embeddings.append(word_emb[None, :])
            if i >= sent_len:
                word_ids.append(self.opt["txt_pad_idx"])
            else:
                word_ids.append(word_id)
        word_embeddings = np.concatenate(word_embeddings, axis=0)
        word_ids = np.array(word_ids, dtype=int)
        
        mot_len = motion.shape[0]
        
        if mot_len > self.opt["amass_window_size"]:
            m_start_idx = np.random.randint(0, mot_len - self.opt["amass_window_size"])
            m_end_idx = m_start_idx + self.opt["amass_window_size"]
            motion = motion[m_start_idx:m_end_idx]
        else:
            motion = ndimage.zoom(motion, (self.opt["amass_window_size"] / mot_len, 1), order=1)
        
        return caption, word_ids, motion
    
    def get_one_aist(self, data):
        return super().get_one_aist(data)
    
""" Read motions in fixed lengths """
class TextAudio2MotionTokenEvalDataset(data.Dataset):
    def __init__(self, opt, amass_split_file, aist_split_file):
        super(TextAudio2MotionTokenEvalDataset, self).__init__()
        self.opt = opt

        dir_path = os.path.dirname(os.path.realpath(__file__))
        valid_test_set = json.load(
            open(os.path.join(dir_path, "ValidTestset_HumanML3D.json"), "r"))
        self.valid_amass_test_set = [t["name"] for t in valid_test_set]

        amass_id_list = []
        aist_id_list = []
        with cs.open(amass_split_file, "r") as f:
            for line in f.readlines():
                amass_id_list.append(line.strip())
        
        with cs.open(aist_split_file, "r") as f:
            for line in f.readlines():
                aist_id_list.append(line.strip())

        amass_data_dict, amass_name_list = self.read_all_amass(amass_id_list)
        aist_data_dict, aist_name_list = self.read_all_aist(aist_id_list)

        self.data_dict = {}
        self.data_dict.update(amass_data_dict)
        self.data_dict.update(aist_data_dict)
        self.name_list = amass_name_list + aist_name_list
        self.modality_list = ["amass" for _ in amass_name_list] + ["aist" for _ in aist_name_list]
        
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        name = self.name_list[item]
        modality = self.modality_list[item]

        if modality == "amass":
            caption, motion = self.get_one_amass(data)
            name = caption
            return modality, name, caption, motion
        elif modality == "aist":
            audio, motion = self.get_one_aist(data)
            name = os.path.split(self.name_list[item])[-1]
            return modality, name, audio, motion
        else:
            raise ValueError
        
    def read_all_amass(self, amass_id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(amass_id_list):
            if name not in self.valid_amass_test_set:
                continue

            try:
                # Read motion
                motion = np.load(pjoin(self.opt["amass_motion_dir"], name + ".npy"))
                
                # Read text
                with cs.open(pjoin(self.opt["text_dir"], name + ".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)
                                data_dict[new_name] = {'text':[text_dict], "motion": motion}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'text': text_data, "motion": motion}
                    new_name_list.append(name)
            except:
                # print(' --- amass pass 2', pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name)))
                pass

        return data_dict, new_name_list
    
    def read_all_aist(self, aist_id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(aist_id_list):
            # print(pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name)))
            try:
                # Read audio
                pickle_data = np.load(pjoin(self.opt["aist_motion_dir"], name + ".npy"), allow_pickle=True).item()
                audio = pickle_data["audio_sequence"]
                
                # Read motion
                motion = pickle_data["motion_smpl"]

                data_dict[name] = {'audio': audio, 'motion': motion}
                
                new_name_list.append(name)

            except:
                pass

        return data_dict, new_name_list
        
    def get_one_amass(self, data):
        text_list = data['text']
        text_data = random.choice(text_list)
        caption, t_tokens = text_data['caption'], text_data['tokens']
        motion = data["motion"]
        
        mot_len = motion.shape[0]
        
        if mot_len > self.opt["amass_window_size"]:
            m_start_idx = np.random.randint(0, mot_len - self.opt["amass_window_size"])
            m_end_idx = m_start_idx + self.opt["amass_window_size"]
            motion = motion[m_start_idx:m_end_idx]
        else:
            motion = ndimage.zoom(motion, (self.opt["amass_window_size"] / mot_len, 1), order=1)
        
        if len(t_tokens) < self.opt["max_text_len"]:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt["max_text_len"] + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt["max_text_len"]]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            
        return caption, motion
    
    def get_one_aist(self, data):
        motion = data["motion"]
        audio = data["audio"]
        
        mot_len = motion.shape[0]
        aud_len = audio.shape[0]
        
        # max_len = min(mot_len, aud_len)
        
        # audio = audio[:max_len]
        # motion = motion[:max_len][::3]
        motion = motion[::3]
        
        return audio, motion

""" Read motions as their tokens (different lengths). """
class TextAudio2MotionTokenEvalDataset_v2(data.Dataset):
    def __init__(self, opt, amass_split_file, aist_split_file):
        super(TextAudio2MotionTokenEvalDataset_v2, self).__init__()
        self.opt = opt

        dir_path = os.path.dirname(os.path.realpath(__file__))
        valid_test_set = json.load(
            open(os.path.join(dir_path, "ValidTestset_HumanML3D.json"), "r"))
        self.valid_amass_test_set = [t["name"] for t in valid_test_set]

        amass_id_list = []
        aist_id_list = []
        with cs.open(amass_split_file, "r") as f:
            for line in f.readlines():
                amass_id_list.append(line.strip())
        
        with cs.open(aist_split_file, "r") as f:
            for line in f.readlines():
                aist_id_list.append(line.strip())

        amass_data_dict, amass_name_list = self.read_all_amass(amass_id_list)
        aist_data_dict, aist_name_list = self.read_all_aist(aist_id_list)

        self.data_dict = {}
        self.data_dict.update(amass_data_dict)
        self.data_dict.update(aist_data_dict)
        self.name_list = amass_name_list + aist_name_list
        self.modality_list = ["amass" for _ in amass_name_list] + ["aist" for _ in aist_name_list]
        print(' --- ', len(self.data_dict))
        print(' --- ', len(self.name_list))

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        name = self.name_list[item]
        modality = self.modality_list[item]

        if modality == "amass":
            caption, m_tokens, sent_len, m_tokens_len, motion = self.get_one_amass(data)
            name = caption
            return modality, name, caption, m_tokens, sent_len, m_tokens_len, motion
        elif modality == "aist":
            audio, m_tokens, aud_len, m_tokens_len, motion = self.get_one_aist(data)
            name = os.path.split(self.name_list[item])[-1]
            return modality, name, audio, m_tokens, aud_len, m_tokens_len, motion
        else:
            raise ValueError

    def read_all_amass(self, amass_id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(amass_id_list):
            if name not in self.valid_amass_test_set:
                continue
                
            try:
                m_token_list = []
                # Read tokens
                with cs.open(pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name)), "r") as f:
                    for line in f.readlines():
                        m_token_list.append(line.strip().split(' '))
                # print('read {:s}'.format(pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name))))

                # Read motion
                motion = np.load(pjoin(self.opt["amass_motion_dir"], name + ".npy"))

                # Read text
                with cs.open(pjoin(self.opt["text_dir"], name + ".txt")) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list = [tokens[int(f_tag*5) : int(to_tag*5)] for tokens in m_token_list]
                                #
                                # if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                #     continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)
                                # new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                # while new_name in data_dict:
                                #     new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                data_dict[new_name] = {'m_token_list': m_token_list,
                                                       'text':[text_dict], 
                                                       "motion": motion}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list, 
                                       'text': text_data, 
                                       "motion": motion}
                    new_name_list.append(name)
            except:
                # print(' --- amass pass 2', pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name)))
                pass

        return data_dict, new_name_list

    def read_all_aist(self, aist_id_list):
        new_name_list = []
        data_dict = {}
        for name in tqdm(aist_id_list):
            # print(pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name)))
            try:
                m_token_list = []
                # Read tokens
                with cs.open(pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name)), "r") as f:
                    for line in f.readlines():
                        m_token_list.append(line.strip().split(' '))
                # print(pjoin(self.opt["tokenizer_name"], "{:s}.txt".format(name)))

                # Read audio
                pickle_data = np.load(pjoin(self.opt["aist_motion_dir"], name + ".npy"), allow_pickle=True).item()
                audio = pickle_data["audio_sequence"]
                
                # Read motion
                motion = pickle_data["motion_smpl"]

                data_dict[name] = {'m_token_list': m_token_list, 
                                   'audio': audio, 
                                   'motion': motion}
                
                new_name_list.append(name)

            except:
                pass

        return data_dict, new_name_list

    def get_one_amass(self, data):
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)
        m_tokens = [int(token) for token in m_tokens]
        text_data = random.choice(text_list)
        caption, t_tokens = text_data['caption'], text_data['tokens']
        motion = data["motion"]

        if len(t_tokens) < self.opt["max_text_len"]:
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
            t_tokens += ['unk/OTHER'] * (self.opt["max_text_len"] + 2 - sent_len)
        else:
            t_tokens = t_tokens[:self.opt["max_text_len"]]
            t_tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(t_tokens)
        
        m_tokens_len = len(m_tokens)
        m_tokens = [self.opt["mot_start_idx"]] + \
                   m_tokens + \
                   [self.opt["mot_end_idx"]] + \
                   [self.opt["mot_pad_idx"]] * (self.opt["max_amass_motion_len"] - len(m_tokens) - 2)
        
        m_tokens = np.array(m_tokens, dtype=int)

        return caption, m_tokens, sent_len, m_tokens_len, motion

    def get_one_aist(self, data):
        m_token_list = data['m_token_list']
        m_tokens = random.choice(m_token_list)
        m_tokens_len = len(m_tokens)
        aud_len = int(3 * self.opt["unit_length"] * m_tokens_len)
        max_aud_len = int(3 * (self.opt["max_aist_motion_len"] - 2) * self.opt["unit_length"])
        
        # Get audio of length associated with motion tokens
        # audio = data["audio"][:int(3 * self.opt["unit_length"] * m_tokens_len)]
        audio = data["audio"]
        aud_len = audio.shape[0]

        # 1. Crop motion tokens and audio for data augmentation, random crop to evaluate
        # m_start_id = np.random.randint(0, m_tokens_len - self.opt["aist_window_size"])
        # m_end_id = m_start_id + self.opt["aist_window_size"]
        # a_start_id = m_start_id * 3 * self.opt["unit_length"]
        # a_end_id = a_start_id + 3 * self.opt["unit_length"] * self.opt["aist_window_size"]

        # 2. Use full sequence to evaluate
        m_start_id = 0
        m_end_id = m_start_id + m_tokens_len
        a_start_id = 0
        a_end_id = a_start_id + 3 * self.opt["unit_length"] * m_tokens_len

        # print(m_start_id, "|", m_end_id, "|", a_start_id, "|", a_end_id, "|", audio.shape)

        # 1. Use fixed length of motion tokens, padding needed if necessary
        # m_tokens = [self.opt["mot_start_idx"]] + \
        #             m_tokens[m_start_id:m_end_id] + \
        #            [self.opt["mot_end_idx"]] + \
        #            [self.opt["mot_pad_idx"]] * (self.opt["max_aist_motion_len"] - self.opt["aist_window_size"] - 2)
        # m_tokens = np.array(m_tokens, dtype=int)

        # 2. Use all motion tokens, no padding needed
        m_tokens = [self.opt["mot_start_idx"]] + \
                    m_tokens[m_start_id:m_end_id] + \
                   [self.opt["mot_end_idx"]]
        m_tokens = np.array(m_tokens, dtype=int)

        # audio = audio[a_start_id:a_end_id]        # DEBUG

        # 1. Use fixed length of motions
        # motion = data["motion"][:, :aud_len]
        # m_start_id_ = m_start_id * self.opt["unit_length"]
        # m_end_id_ = m_start_id_ + self.opt["aist_window_size"] * self.opt["unit_length"]
        # motion = motion[::3]
        # motion = motion[m_start_id_:m_end_id_]

        # 2. Use full length of motion
        # print(' ---> ', data["motion"].shape)
        motion = data["motion"][:a_end_id][::3]
        # print(' ---> ', motion.shape)

        return audio, m_tokens, audio.shape[0], m_tokens.shape[0], motion

""" Define Dataloaders 
"""
def get_training_tokenizer_dataloader(opt, meta_dir=None):
    
    amass_train_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'train.txt')
    aist_train_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'train.txt')
    train_dataset = MotionTokenizerDataset(opt["data"]["dataset"], amass_train_split_file, aist_train_split_file, meta_dir)
    train_loader = DataLoader(train_dataset, 
                              batch_size=opt["data"]["loader"]["batch_size"], 
                              drop_last=True, 
                              num_workers=opt["data"]["loader"]["workers"], 
                              shuffle=True, 
                              pin_memory=True)
    return train_loader, train_dataset

def get_vald_tokenizer_dataloader(opt, meta_dir=None):

    amass_vald_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'val.txt')
    aist_vald_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'val.txt')
    train_dataset = MotionTokenizerDataset(opt["data"]["dataset"], amass_vald_split_file, aist_vald_split_file, meta_dir)
    train_loader = DataLoader(train_dataset, 
                              batch_size=opt["data"]["loader"]["batch_size"], 
                              drop_last=True, 
                              num_workers=opt["data"]["loader"]["workers"], 
                              shuffle=True, 
                              pin_memory=True)
    return train_loader, train_dataset

def get_training_vae_dataloader(opt, meta_dir=None):
    amass_train_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'train.txt')
    aist_train_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'train.txt')
    train_dataset = MotionVAEDataset(opt["data"]["dataset"], amass_train_split_file, aist_train_split_file)
    train_loader = DataLoader(train_dataset, 
                              batch_size=opt["data"]["loader"]["batch_size"], 
                              drop_last=True, 
                              num_workers=opt["data"]["loader"]["workers"], 
                              shuffle=True, 
                              pin_memory=True)
    return train_loader, train_dataset

def get_vald_vae_dataloader(opt, meta_dir=None):
    amass_vald_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'val.txt')
    aist_vald_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'val.txt')
    train_dataset = MotionVAEDataset(opt["data"]["dataset"], amass_vald_split_file, aist_vald_split_file)
    train_loader = DataLoader(train_dataset, 
                              batch_size=opt["data"]["loader"]["batch_size"], 
                              drop_last=True, 
                              num_workers=opt["data"]["loader"]["workers"], 
                              shuffle=True, 
                              pin_memory=True)
    return train_loader, train_dataset

def get_training_diffae_dataloader(opt, meta_dir=None):
    amass_train_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'train.txt')
    aist_train_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'train.txt')
    train_dataset = MotionDiffusionAEDataset(opt["data"]["dataset"], amass_train_split_file, aist_train_split_file)
    train_loader = DataLoader(train_dataset, 
                              batch_size=opt["data"]["loader"]["batch_size"], 
                              drop_last=True, 
                              num_workers=opt["data"]["loader"]["workers"], 
                              shuffle=True, 
                              pin_memory=True)
    return train_loader, train_dataset

def get_vald_diffae_dataloader(opt, meta_dir=None):
    amass_train_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'val.txt')
    aist_train_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'val.txt')
    train_dataset = MotionDiffusionAEDataset(opt["data"]["dataset"], amass_train_split_file, aist_train_split_file)
    train_loader = DataLoader(train_dataset, 
                              batch_size=opt["data"]["loader"]["batch_size"], 
                              drop_last=True, 
                              num_workers=opt["data"]["loader"]["workers"], 
                              shuffle=True, 
                              pin_memory=True)
    return train_loader, train_dataset

def get_all_tokenizer_dataloader(opt, meta_dir=None):

    # amass_split_file = [os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'train.txt'), 
    #                     os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'val.txt')]
    # aist_split_file = [os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'train.txt'), 
    #                    os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'val.txt')]
    amass_split_file = [os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'all.txt')]
    aist_split_file = [os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'train.txt'), 
                       os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'val.txt')]
    dataset = TokenizerDataset(opt["data"]["dataset"], amass_split_file, aist_split_file, meta_dir)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, pin_memory=True)

    return loader, dataset

def get_training_ta2m_dataloader(opt):

    amass_train_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'train.txt')
    aist_train_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'train.txt')

    train_dataset = TextAudio2MotionTokenDataset(opt["data"]["dataset"], amass_train_split_file, aist_train_split_file)

    train_loader = DataLoader(train_dataset, 
                              batch_size=opt["data"]["loader"]["batch_size"], 
                              drop_last=True, 
                              num_workers=opt["data"]["loader"]["workers"], 
                              shuffle=True, 
                              pin_memory=True)

    return train_loader, train_dataset

def get_vald_ta2m_dataloader(opt):
    amass_vald_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'val.txt')
    aist_vald_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'val.txt')

    val_dataset = TextAudio2MotionTokenDataset(opt["data"]["dataset"], amass_vald_split_file, aist_vald_split_file)

    val_loader = DataLoader(val_dataset, 
                            batch_size=opt["data"]["loader"]["batch_size"], 
                            drop_last=True, 
                            num_workers=opt["data"]["loader"]["workers"], 
                            shuffle=True, 
                            pin_memory=True)

    return val_loader, val_dataset

def get_training_ta2m_dataloader_v2(opt):

    amass_train_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'train.txt')
    aist_train_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'train.txt')

    train_dataset = TextAudio2MotionTokenDataset_v2(opt["data"]["dataset"], amass_train_split_file, aist_train_split_file)

    train_loader = DataLoader(train_dataset, 
                              batch_size=opt["data"]["loader"]["batch_size"], 
                              drop_last=True, 
                              num_workers=opt["data"]["loader"]["workers"], 
                              shuffle=True, 
                              pin_memory=True)

    return train_loader, train_dataset

def get_vald_ta2m_dataloader_v2(opt):
    amass_vald_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'val.txt')
    aist_vald_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'val.txt')

    val_dataset = TextAudio2MotionTokenDataset_v2(opt["data"]["dataset"], amass_vald_split_file, aist_vald_split_file)

    val_loader = DataLoader(val_dataset, 
                            batch_size=opt["data"]["loader"]["batch_size"], 
                            drop_last=True, 
                            num_workers=opt["data"]["loader"]["workers"], 
                            shuffle=True, 
                            pin_memory=True)

    return val_loader, val_dataset

def get_training_ta2m_cycle_dataloader(opt):
    cur_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    glob_path = pjoin(cur_path, 'perception', 'glove')
    w_vectorizer = WordVectorizerV2(glob_path, 'our_vab')
    
    amass_train_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'train.txt')
    aist_train_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'train.txt')
    train_dataset = TextAudio2MotionTokenDataset_cycle(opt=opt["data"]["dataset"], 
                                                       amass_split_file=amass_train_split_file, 
                                                       aist_split_file=aist_train_split_file, 
                                                       w_vectorizer=w_vectorizer)
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=opt["data"]["loader"]["batch_size"], 
                              drop_last=True, 
                              num_workers=opt["data"]["loader"]["workers"], 
                              shuffle=True, 
                              pin_memory=True)
    return train_loader, train_dataset

def get_vald_ta2m_cycle_dataloader(opt):
    import copy
    cur_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    glob_path = pjoin(cur_path, 'perception', 'glove')
    w_vectorizer = WordVectorizerV2(glob_path, 'our_vab')
    
    opt_copy = copy.deepcopy(opt)
    opt_copy["data"]["dataset"]["times"] = 1
    
    amass_vald_split_file = os.path.join(opt_copy["data"]["dataset"]["amass_base_dir"], "val.txt")
    aist_vald_split_file = os.path.join(opt_copy["data"]["dataset"]["aist_base_dir"], 'val.txt')
    vald_dataset = TextAudio2MotionTokenDataset_cycle(opt=opt_copy["data"]["dataset"], 
                                                      amass_split_file=amass_vald_split_file, 
                                                      aist_split_file=aist_vald_split_file, 
                                                      w_vectorizer=w_vectorizer)
    vald_loader = DataLoader(dataset=vald_dataset, 
                             batch_size=opt_copy["data"]["loader"]["batch_size"], 
                             drop_last=True, 
                             num_workers=opt_copy["data"]["loader"]["workers"], 
                             shuffle=True, 
                             pin_memory=True)
    return vald_dataset, vald_loader
   
def get_eval_ta2m_dataloader(opt):
    amass_eval_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'test.txt')
    # aist_eval_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'val.txt')
    aist_eval_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'test_all.txt')

    eval_dataset = TextAudio2MotionTokenEvalDataset(opt["data"]["dataset"], amass_eval_split_file, aist_eval_split_file)
    eval_loader = DataLoader(eval_dataset, 
                             batch_size=opt["data"]["loader"]["batch_size"], 
                             num_workers=1, 
                             shuffle=False, 
                             pin_memory=True)

    return eval_loader, eval_dataset

def get_eval_ta2m_dataloader_v2(opt):
    amass_eval_split_file = os.path.join(opt["data"]["dataset"]["amass_base_dir"], 'test.txt')
    # aist_eval_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'val.txt')
    aist_eval_split_file = os.path.join(opt["data"]["dataset"]["aist_base_dir"], 'test_all.txt')

    eval_dataset = TextAudio2MotionTokenEvalDataset_v2(opt["data"]["dataset"], amass_eval_split_file, aist_eval_split_file)
    eval_loader = DataLoader(eval_dataset, 
                             batch_size=opt["data"]["loader"]["batch_size"], 
                             num_workers=1, 
                             shuffle=False, 
                             pin_memory=True)

    return eval_loader, eval_dataset
        
