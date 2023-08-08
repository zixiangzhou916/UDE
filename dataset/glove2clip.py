import os
os.environ["CONDA_DLL_SEARCH_MODIFICATION_ENABLE"]="1"
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
from dataset.perception.word_vectorizer import WordVectorizerV2
from networks.ImageCLIP import clip
from tqdm import tqdm
from dataset.perception.dataset import get_training_motion2text_dataloader

def get_glove2clip_transformation():
    w_vectorizer = WordVectorizerV2("dataset/perception/glove", "our_vab")
    num_glove_tokens = len(w_vectorizer.idx2word)
    
    token_to_glove = {}
    for idx in tqdm(range(num_glove_tokens+1)):
        text = w_vectorizer.itos(idx)
        token_to_glove[idx] = text
        
    g_text = [t for _, t in token_to_glove.items()]
    print(len(set(g_text)))
    
    glove_to_clip = {}
    for idx, text in tqdm(token_to_glove.items()):
        clip_token = clip.tokenize([text], truncate=True)
        if "sos" == text:
            glove_to_clip[text] = clip_token[0, 0].item()
        elif "eos" == text:
            glove_to_clip[text] = clip_token[0, 2].item()
        elif "pad" == text:
            glove_to_clip[text] = 0
        else:
            glove_to_clip[text] = clip_token[0, 1].item()
        
    transformation = []
    clip_idxs = []
    for idx, text in tqdm(token_to_glove.items()):
        clip_idx = glove_to_clip[text]
        transformation.append([idx, clip_idx])
        clip_idxs.append(clip_idx)    
        
    print(len(set(clip_idxs)))
    
    # Make sparse
    indices = np.asarray(transformation)[:, 1]
    
    output = {
        "token_to_glove": token_to_glove, 
        "glove_to_clip": glove_to_clip, 
        "indices": indices
    }
    
    return output

if __name__ == "__main__":
    import yaml
    with open("configs/perception/config_m2t_gpt_v1.yaml", "r") as f:
        opt = yaml.safe_load(f)
    _, dataset = get_training_motion2text_dataloader(opt=opt, meta_dir=None)
    
    g2c_transformation = get_glove2clip_transformation()
    np.save("glove_to_clip_transformation.npy", g2c_transformation)
    indices = torch.LongTensor(g2c_transformation["indices"])
    
    clip_model, _ = clip.load(name="ViT-B/32", jit=False, 
                              download_root="networks/ImageCLIP/models/ViT-B-32.pt")
    clip_model.eval()
    
    for idx in range(dataset.__len__()):
        batch = dataset.__getitem__(idx)
        word_embeddings, word_ids, caption, sent_len, motion = batch
        word_ids = torch.LongTensor(word_ids)
        
        token_1 = clip.tokenize([caption], truncate=True)
        token_2 = indices[word_ids]
        pad_2 = torch.zeros(77 - token_2.size(0)).long()
        token_2 = torch.cat((token_2, pad_2), dim=0)[None]
        
        with torch.no_grad():
            embd_1 = clip_model.encode_text(token_1).float()
            embd_2 = clip_model.encode_text(token_2).float()
            cos = torch.nn.functional.cosine_similarity(embd_1, embd_2, dim=-1)
            print(cos)