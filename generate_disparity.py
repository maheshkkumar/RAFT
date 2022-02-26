import argparse
import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from core.utils.frame_utils import readFlow


def min_max(img):
    dmin, dmax = img.min(), img.max()
    img = (img - dmin) / (dmax - dmin)
    return img

def flow_to_disp(args):
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # read LR flow files
    lr_flow_list = sorted(glob(args.flow_path + '/*left_right.flo'))
    print(f"Processing {len(lr_flow_list)} LR flow files")
    
    for lr_flow_path in tqdm(lr_flow_list, total=len(lr_flow_list)):
        lr_flow = readFlow(lr_flow_path)
        
        disparity = np.clip(1 - (min_max(lr_flow[...,0])), 0.01, 1)
        
        # convert disparty to 16-bit int
        disparity = disparity * (2**16 - 1)
        
        # np to PIL 
        disparity = Image.fromarray(disparity.astype(np.uint16))
        
        base_name = os.path.basename(lr_flow_path).split('.')[0].replace('left_right', 'disparity')
        output_path = os.path.join(args.output_path, f"{base_name}.png")
        
        # save image
        disparity.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flow_path', type=str, help='Path to input flow data with L->R flow')
    parser.add_argument('--output_path', type=str, help='Path to output folder')
    
    args = parser.parse_args()
    flow_to_disp(args)
    
