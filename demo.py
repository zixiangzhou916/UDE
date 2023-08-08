import os
import argparse
import yaml
from modules.evaluator import UDEEvaluator

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_ude.yaml')
    parser.add_argument('--eval_folder', type=str, default='./results/eval', help='path of evaluation folder')
    parser.add_argument('--eval_name', type=str, default='ude', help='name of the evaluation')
    parser.add_argument('--repeat_times', type=int, default='3', help='number of repeat times per sample')
    parser.add_argument('--repeat_times_dmd', type=int, default='3', help='number of repeat times per sample (DMD)')
    parser.add_argument('--add_noise', type=str2bool, default=False, help='whether to inject random noise')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='scale of noise, if noise distribution is uniform, then this argument is the boundary, if the noise distribution is normal, then this argument is the std')
    parser.add_argument('--dist_type', type=str, default="uniform", help='distribution of noise')
    parser.add_argument('--latent_scale', type=float, default=1.0, help='scale factor to VAE latent')
    parser.add_argument('--eval_mode', type=str, default='t,a', help='evaluation mode, t means text-to-motion, a means audio-to-motion')
    parser.add_argument('--audio_seg_len', type=int, default=64, help='length of audio segment to decode')
    parser.add_argument('--use_dmd', type=str2bool, default=True, help='whether to use DiffusionMotionDecoder or VQDecoder')
    parser.add_argument('--condition_type', type=str, default='audio', help='type of condition, choose from [text, audio]')
    parser.add_argument('--condition_input', type=str, default='sample_data/a2m', help='input path of condition info')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    Agent = UDEEvaluator(args, config)
    if args.condition_type == "text":
        Agent.generate_t2m(args.condition_input)
    elif args.condition_type == "audio":
        Agent.generate_a2m(args.condition_input)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)
