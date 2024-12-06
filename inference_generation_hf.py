"""
ScaMo: Motion Generation Model Inference Script

This script provides functionality for generating human motions from text descriptions using 
a pretrained ScaMo model. It uses a VQ-VAE architecture combined with a LLaMA-based transformer
for text-to-motion generation.

Example usage:
    python inference_generation_hf.py --nb-code 65536 --quantizer FSQ --pretrained_llama 3B --text_encode flan-t5-xl
"""

import torch
import options.option_transformer as option_trans
import models.t2m_trans as trans
import clip
import numpy as np
import random
import models.vqvae as vqvae
import os
from models.lit_llama.model_hf import LLaMAHF, LLaMAHFConfig
from transformers import T5EncoderModel, T5Tokenizer
from utils.quaternion import *
from visualization.plot_3d_global import plot_3d_motion
import imageio
import sys

def inv_transform(data, mean, std):
    """Inverse transform normalized data back to original scale"""
    return data * std + mean

def recover_root_rot_pos(data):
    """
    Recover root rotation and position from motion data
    
    Args:
        data: Motion data tensor
    Returns:
        r_rot_quat: Root rotation quaternion
        r_pos: Root position
    """
    # Extract rotation velocity
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    
    # Get Y-axis rotation from rotation velocity
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    # Convert to quaternion
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device).to(data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    # Get root position
    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device).to(data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    
    # Add Y-axis rotation to root position
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    """
    Recover full motion from rotation-invariant coordinates
    
    Args:
        data: Motion data in rotation-invariant coordinates
        joints_num: Number of joints
    Returns:
        positions: Full motion positions
    """
    # Get root rotation and position
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    
    # Extract joint positions
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    # Add Y-axis rotation to local joints
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    # Add root XZ to joints
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    # Concatenate root and joints
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

@torch.no_grad()
def plot(pred_pose_denorm, dataname):
    """
    Plot predicted motion sequence
    
    Args:
        pred_pose_denorm: Denormalized predicted pose
        dataname: Dataset name ('t2m' or 'kit')
    Returns:
        pred_xyz: 3D joint positions
        img: Visualization frames
    """
    pred_xyz = recover_from_ric(pred_pose_denorm, joints_num=22 if dataname == 't2m' else 21).detach().cpu().numpy()[0]
    img = plot_3d_motion([pred_xyz, None, None])
    return pred_xyz, img


if __name__ == '__main__':
    # Set compute device
    comp_device = torch.device('cuda')

    # Initialize text prompt
    text_data = ['A man is walking forward.']
    
    # Get model configuration
    args = option_trans.get_args_parser()

    # Load text encoder model (CLIP or T5)
    if args.text_encode == 'clip':
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=comp_device, jit=False)
        clip.model.convert_weights(clip_model)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
            
    elif args.text_encode == 'flan-t5-xl':
        tokenizer = T5Tokenizer.from_pretrained('pretrained/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001', local_files_only=True)
        text_encoder = T5EncoderModel.from_pretrained('pretrained/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001', local_files_only=True).to(device=comp_device)
        clip_model = (tokenizer, text_encoder)
        clip_model[1].eval()
        for p in clip_model[1].parameters():
            p.requires_grad = False
        args.clip_dim = 2048
        print(f'Flan-t5-xl loaded')
        
    elif args.text_encode == 'flan-t5-xxl':
        tokenizer = T5Tokenizer.from_pretrained('pretrained/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True)
        text_encoder = T5EncoderModel.from_pretrained('pretrained/models--google--flan-t5-xxl/snapshots/ae7c9136adc7555eeccc78cdd960dfd60fb346ce', local_files_only=True).to(device=comp_device)
        clip_model = (tokenizer, text_encoder)
        clip_model[1].eval()
        for p in clip_model[1].parameters():
            p.requires_grad = False
        args.clip_dim = 4096
        print(f'Flan-t5-xxl loaded')
        
    else:
        raise ValueError(f'Unknown text encoder: {args.text_encode}')

    # Load VQ-VAE model
    net = vqvae.HumanVQVAE(args,
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate)

    ckpt = torch.load(f'pretrained/ScaMo_3B/{args.nb_code}_FSQ.pth', map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.to(comp_device)
    print('Load VQVAE model successfully!')

    # Configure and load transformer model
    args.nb_code = net.vqvae.quantizer.codebook_size
    config = LLaMAHFConfig.from_name(args.pretrained_llama)
    config.block_size = args.block_size
    config.vocab_size = args.nb_code + 2
    config.clip_dim = args.clip_dim
    config.tie_weights = args.tie_weights
    print(config)
    
    trans_encoder = LLaMAHF(config)
    ckpt = torch.load(f'pretrained/ScaMo_{args.pretrained_llama}/ScaMo_net_{args.pretrained_llama}.pth', map_location='cpu')
    ckpt = {k.replace('module.', ''): v for k, v in ckpt['trans'].items()}
    trans_encoder.load_state_dict(ckpt, strict=True)
    trans_encoder.eval()
    trans_encoder.to(comp_device)
    print('Load transformer model successfully!')

    # Main inference loop
    while True:
        # Get text input
        input_text = input('Input text: ')
        clip_text = input_text

        # Encode text based on selected encoder
        if args.text_encode == 'clip':
            text = clip.tokenize(clip_text, truncate=True).to(comp_device)
            feat_clip_text = clip_model.encode_text(text).float()
            feat_clip_text = feat_clip_text.unsqueeze(1)
            y_mask = torch.ones((feat_clip_text.shape[0], feat_clip_text.shape[1])).to(comp_device)
            assert args.text_sum_way is None
            
        elif args.text_encode == 'flan-t5-xxl':
            tokenizer, text_encoder = clip_model
            cap_inputs = tokenizer(clip_text, padding=True, truncation=True, return_tensors="pt")
            y_mask = cap_inputs.attention_mask.to(device=comp_device)
            feat_clip_text = text_encoder(
                input_ids=cap_inputs.input_ids.to(comp_device), 
                attention_mask=cap_inputs.attention_mask.to(comp_device), 
                output_hidden_states=False
            ).last_hidden_state
            
        elif args.text_encode == 'flan-t5-xl':
            tokenizer, text_encoder = clip_model
            cap_inputs = tokenizer(clip_text, padding=True, truncation=True, return_tensors="pt")
            y_mask = cap_inputs.attention_mask.to(device=comp_device)
            feat_clip_text = text_encoder(
                input_ids=cap_inputs.input_ids.to(comp_device), 
                attention_mask=cap_inputs.attention_mask.to(comp_device),
                output_hidden_states=False
            ).last_hidden_state
            
        else:
            raise NotImplementedError

        # Truncate long sequences
        if feat_clip_text.shape[1] > 150:
            feat_clip_text = feat_clip_text[:, :150, :]
            y_mask = y_mask[:, :150]

        # Apply text pooling if specified
        if args.text_sum_way == 'cls':
            feat_clip_text = feat_clip_text[:, 0, :]
        elif args.text_sum_way == 'mean':
            feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1) / y_mask.sum(dim=1, keepdim=True)
        elif args.text_sum_way == 'sum':
            feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1)

        # Generate motion
        index_motion = trans_encoder.sample(feat_clip_text, y_mask, if_categorial=False)
        print(index_motion)

        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)
        
        # Decode motion
        pred_pose = net.forward_decoder(index_motion)

        # Load mean and std for denormalization
        mean = np.load('pretrained/ScaMo_3B/mean.npy')
        std = np.load('pretrained/ScaMo_3B/std.npy')

        # Denormalize and save results
        pred_pose = inv_transform(pred_pose.detach().cpu().numpy(), mean, std)
        np.save('output/predict.npy', pred_pose[0])
        print('save pose!')
        
        # Generate visualization
        generated_pose, img = plot(torch.from_numpy(pred_pose).cuda(), args.dataname)
        
        # Save visualization
        short_name = clip_text[:50].strip() + '...' if len(clip_text) > 50 else clip_text
        imageio.mimsave(os.path.join('output', f'{short_name}.gif'), np.array(img), fps=20)
        
        print("Inference done!")
