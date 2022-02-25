import argparse
import glob
import os
from distutils.archive_util import make_archive

import cv2
import numpy as np
import torch
from PIL import Image

from core.raft import RAFT
from core.utils import flow_viz
from core.utils.frame_utils import writeFlow
from core.utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    
    os.makedirs(args.output_path, exist_ok=True)

    with torch.no_grad():
        
        left_images = sorted(glob.glob(os.path.join(args.input_path, 'left', '*.png')))
        right_images = sorted(glob.glob(os.path.join(args.input_path, 'right', '*.png')))
        
        assert os.path.basename(left_images[0].replace('left', 'right')) == os.path.basename(right_images[0])
        
        for imfile1, imfile2 in zip(left_images, right_images):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, lr_flow_up = model(image1, image2, iters=20, test_mode=True)
            _, rl_flow_up = model(image2, image1, iters=20, test_mode=True)
            
            lr_flow_up = lr_flow_up[0].permute(1,2,0).cpu().numpy()
            rl_flow_up = rl_flow_up[0].permute(1,2,0).cpu().numpy()
            
            lr_flow = os.path.basename(imfile1).replace('left', 'left_right').replace('.png', '.flo')
            rl_flow = os.path.basename(imfile2).replace('right', 'right_left').replace('.png', '.flo')
            
            lr_out_path = os.path.join(args.output_path, lr_flow)
            rl_out_path = os.path.join(args.output_path, rl_flow)
            
            writeFlow(lr_flow_up, lr_out_path)
            writeFlow(rl_flow_up, rl_out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--input_path', help="input dataset path")
    parser.add_argument('--output_path', help="output dataset path")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
