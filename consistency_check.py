import argparse
import os
from glob import glob

import numpy as np
import torch
import torch.nn
from PIL import Image
from tqdm import tqdm

from core.utils.frame_utils import readFlow


# Source: https://github.com/facebookresearch/consistent_depth/blob/e2c9b724d3221aa7c0bf89aa9449ae33b418d943/utils/consistency.py
def sample(data, uv):
    """Sample data (H, W, <C>) by uv (H, W, 2) (in pixels). """
    shape = data.shape
    # data from (H, W, <C>) to (1, C, H, W)
    data = data.reshape(data.shape[:2] + (-1,))
    data = torch.tensor(data).permute(2, 0, 1)[None, ...]
    # (H, W, 2) -> (1, H, W, 2)
    uv = torch.tensor(uv)[None, ...]

    H, W = shape[:2]
    # grid needs to be in [-1, 1] and (B, H, W, 2)
    size = torch.tensor((W, H), dtype=uv.dtype).view(1, 1, 1, -1)
    grid = (2 * uv / size - 1).to(data.dtype)
    tensor = torch.nn.functional.grid_sample(data, grid, padding_mode="border")
    # from (1, C, H, W) to (H, W, <C>)
    return tensor.permute(0, 2, 3, 1).reshape(shape).numpy()


def sse(x, y, axis=-1):
    """Sum of suqare error"""
    d = x - y
    return np.sum(d * d, axis=axis)


def consistency_mask(im_ref, im_tgt, flow, threshold, diff_func=sse):
    H, W = im_ref.shape[:2]
    im_ref = im_ref.reshape(H, W, -1)
    im_tgt = im_tgt.reshape(H, W, -1)
    x, y = np.arange(W), np.arange(H)
    X, Y = np.meshgrid(x, y)
    u, v = flow[..., 0], flow[..., 1]
    idx_x, idx_y = u + X, v + Y

    # first constrain to within the image
    mask = np.all(
        np.stack((idx_x >= 0, idx_x <= W - 1, 0 <= idx_y, idx_y <= H - 1), axis=-1),
        axis=-1,
    )

    im_tgt_to_ref = sample(im_tgt, np.stack((idx_x, idx_y), axis=-1))

    mask = np.logical_and(mask, diff_func(im_ref, im_tgt_to_ref) < threshold)
    return mask

def check_consistency(args):
    
    # LR, RL flow
    lr_flow_list = sorted(glob(args.lr_flow_dir + '/*left_right.flo'))
    rl_flow_list = sorted(glob(args.rl_flow_dir + '/*right_left.flo'))
    
    assert len(lr_flow_list) == len(rl_flow_list)
    
    flow_pairs_list = [[lr, rl] for lr, rl in zip(lr_flow_list, rl_flow_list)]
    
    print(f"Processing {len(flow_pairs_list)} pairs of flow files")
    
    for flow_pair in tqdm(flow_pairs_list, total=len(flow_pairs_list)):
        lr_flow = readFlow(flow_pair[0])
        rl_flow = readFlow(flow_pair[1])
        
        mask = consistency_mask(im_ref=lr_flow, im_tgt=-rl_flow, flow=lr_flow, threshold=args.threshold ** 2)
        
        h, w = mask.shape[:2]
        total_pixels = h * w
        if np.sum(mask) / total_pixels < args.valid_threshold:
            print(f"{flow_pair[0]} and {flow_pair[1]} are have less than {args.valid_threshold} valid pixels")
            continue
        
        # save mask
        img_name = os.path.basename(flow_pair[0]).split('.')[0].replace('left_right', 'mask')
        output_path = os.path.join(args.output_dir, f"{img_name}.png")
        
        mask = Image.fromarray(mask.astype(np.uint8) * 255).convert('L')
        mask.save(output_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, help='Path to input flow data with L->R and R->L flow')
    parser.add_argument('--output_path', type=str, help='Path to output mask')
    parser.add_argument('--threshold', type=float, help='Horizontal direction threshold for flow consistency check, default 1 pixels', default=1)
    parser.add_argument('--valid_threshold', type=float, help='Percentage of pixels that must be consistent, default 0.7', default=0.7)
    
    args = parser.parse_args()
    check_consistency(args)
