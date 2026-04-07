import numpy as np
import random

import importlib
from pathlib import Path
import os

from backbone import sampler, draw, convolution

importlib.reload(sampler)
importlib.reload(draw)
importlib.reload(convolution)

from matplotlib import pyplot as plt

import torch
import torch.nn as nn

kernel_tensor_1, angles = convolution.make_gabor_kernels(ksize=7, sigma=2, lam=4)
kernel_tensor_2, _ = convolution.make_gabor_kernels(ksize=13, sigma=3, lam=7)
kernel_tensor_3, _ = convolution.make_gabor_kernels(ksize=31, sigma=5, lam=15)
kernel_tensor_4, _ = convolution.make_gabor_kernels(ksize=51, sigma=10, lam=25)

outdir = Path(os.getcwd()) / 'outputs'
os.makedirs(outdir, exist_ok=True)

radius = 0.02
thresh = 1e-3

C, H = sampler.draw_positions(radius, sampler.bezier_curve(Ps), thresh=thresh)
D, _ = sampler.draw_positions(radius, sampler.box(), exclusions=C, thresh=thresh)

l=0.025
w=0.005
pfunc = lambda z,h: draw.segment(z, h, l, w)

N = 512
If, Tf = draw.generate_image(C, H, N=N, pfunc=pfunc)
Ig, Tg = draw.generate_image(D, N=N, pfunc=pfunc)

I = If + Ig

kernel_tensor_1 = kernel_tensor_1.to(torch.float32)
kernel_tensor_2 = kernel_tensor_2.to(torch.float32)
kernel_tensor_3 = kernel_tensor_3.to(torch.float32)
kernel_tensor_4 = kernel_tensor_4.to(torch.float32)
plt.imshow(I, aspect='equal',origin= "lower", cmap='binary')

for i in range(50 ):

    # --- Convert Ig to torch tensor ---
    img_np = Ig
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # --- First convolution ---
    conv1 = nn.functional.conv2d(
        img_tensor,
        kernel_tensor_2,
        stride=1,
        padding=3
    )

    # --- Second convolution ---
    conv2 = nn.functional.conv2d(
        conv1,
        kernel_tensor_3,
        stride=1,
        padding=15,
        groups=20
    )

        # --- Third convolution ---
    conv3 = nn.functional.conv2d(
        conv2,
        kernel_tensor_4,
        stride=1,
        padding=15,
        groups=20
    )

    # --- Compute metrics of current Ig ---
    curr_mean = (conv3**2).mean(dim=[2, 3]).sum().item()
    curr_var = conv3.var(dim=[2, 3], unbiased=False).sum().item()

    # --- Build updated candidate image ---
    updated_img = img_np.copy()
    cluster_list = convolution.clusters(conv2)

    indices = random.sample(range(26), 13)
    
    for j in indices:
        val, ch_main, px, py = cluster_list[j]

        ch_weak, weak_val = convolution.weakest_channel(conv2, px, py)
        cx, cy = convolution.find_closest(img_np, px, py)
        connected = convolution.find_connected(img_np, cx, cy)
        outline = convolution.bounding_square(connected)
        x0, y0 = outline[0]
        x1, y1 = outline[1]
        side = outline[2]

        if x0 >= 0 and y0 >= 0 and x1 < 512 and y1 < 512:
            angle = angles[ch_weak] + random.uniform(-1, 1) * 0.1
            patch = convolution.draw_segment(angle, side)
            updated_img[y0:y1, x0:x1] = patch

    # --- Recompute conv2d on the candidate image ---
    cand_tensor = torch.tensor(updated_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    cand_conv1 = nn.functional.conv2d(
        cand_tensor, kernel_tensor_2,
        stride=1, padding=3
    )

    cand_conv2 = nn.functional.conv2d(
        cand_conv1, kernel_tensor_3,
        stride=1, padding=15,
        groups=20
    )
    cand_conv3 = nn.functional.conv2d(
        cand_conv2, kernel_tensor_4,
        stride=1, padding=15,
        groups=20
    )
    # Metrics of candidate image
    cand_mean = (cand_conv3**2).mean(dim=[2, 3]).sum().item()
    cand_var = cand_conv3.var(dim=[2, 3], unbiased=False).sum().item()

    # --- UPDATE ONLY IF IMPROVED ---
    if cand_mean < curr_mean and cand_var < curr_var:
        Ig = updated_img
        print(f"Iter {i+1}: UPDATE | mean {(curr_mean/6):.3f} → {(cand_mean/6):.3f}, var {(curr_var/6):.3f} → {(cand_var/6):.3f}")
    else:
        # No update, but NO FREEZE — next iteration still proceeds
        print(f"Iter {i+1}: NO UPDATE | mean={(curr_mean/6):.3f}, var={(curr_var/6):.3f}")
