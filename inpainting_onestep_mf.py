"""
One-step MeanFlow Inpainting Script
"""

import os
import re
import click
import pickle
import random
import zipfile
import numpy as np
import torch
import PIL.Image
import tqdm
import dnnlib

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32))
            for seed in seeds
        ]

    def randn(self, size, **kwargs):
        return torch.stack([
            torch.randn(size[1:], generator=g, **kwargs)
            for g in self.generators
        ])

    def randint(self, *args, size, **kwargs):
        return torch.stack([
            torch.randint(*args, size=size[1:], generator=g, **kwargs)
            for g in self.generators
        ])


def parse_int_list(s):
    if isinstance(s, list):
        return s
    result = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            result.extend(range(int(a), int(b)+1))
        else:
            result.append(int(part))
    return result


def load_random_cifar(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = [f for f in z.namelist() if f.endswith('.png')]
        selected = random.choice(files)
        with z.open(selected) as f:
            return np.array(PIL.Image.open(f).convert("RGB"))


def generate_random_mask(h, w, size):
    mask = np.zeros((h, w), dtype=np.uint8)
    top = random.randint(0, h-size)
    left = random.randint(0, w-size)
    mask[top:top+size, left:left+size] = 255
    return mask


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

@click.command()
@click.option('--network', required=True, help='Network snapshot (.pkl)')
@click.option('--outdir', required=True, help='Output directory')
@click.option('--seeds', default='0', help='Seeds (e.g. 0 or 0-10)')
@click.option('--class_idx', '--class', default=None, type=int)
@click.option('--image', default=None, help='Path to image')
@click.option('--mask', default=None, help='Path to mask')
@click.option('--cifar_zip', default=None, help='Fallback CIFAR zip')
@click.option('--mask_size', default=12, type=int)
@click.option('--device', default='cuda')
def main(network, outdir, seeds, class_idx,
         image, mask, cifar_zip, mask_size, device):

    device = torch.device(device)
    os.makedirs(outdir, exist_ok=True)

    seeds = parse_int_list(seeds)

    # ---------------------------------------------------------
    # Load network
    # ---------------------------------------------------------

    print("Loading network...")
    with dnnlib.util.open_url(network) as f:
        net = pickle.load(f)['ema'].to(device)

    H = net.img_resolution
    W = net.img_resolution

    # ---------------------------------------------------------
    # Load or sample original image
    # ---------------------------------------------------------

    if image is not None:
        original = np.array(
            PIL.Image.open(image).convert("RGB").resize((W, H))
        )
    elif cifar_zip is not None:
        original = load_random_cifar(cifar_zip)
    else:
        raise ValueError("Provide --image or --cifar_zip")

    PIL.Image.fromarray(original).save(os.path.join(outdir, "original.png"))

    # ---------------------------------------------------------
    # Load or generate mask
    # ---------------------------------------------------------

    if mask is not None:
        mask_np = np.array(
            PIL.Image.open(mask).convert("L").resize((W, H))
        )
    else:
        mask_np = generate_random_mask(H, W, mask_size)

    PIL.Image.fromarray(mask_np).save(os.path.join(outdir, "mask.png"))

    mask_tensor = torch.from_numpy(mask_np).float() / 255.0
    mask_tensor = (mask_tensor > 0.5).float()
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)
    mask_tensor = mask_tensor.expand(1, net.img_channels, H, W)

    # Save masked image
    mask3 = np.stack([mask_np]*3, axis=-1)
    masked_np = original.copy()
    masked_np[mask3 == 255] = 0
    PIL.Image.fromarray(masked_np).save(os.path.join(outdir, "masked.png"))

    # Normalize original
    x_orig = torch.from_numpy(original).float() / 127.5 - 1
    x_orig = x_orig.permute(2,0,1).unsqueeze(0).to(device)

    # ---------------------------------------------------------
    # Generate images
    # ---------------------------------------------------------

    print("Running MeanFlow inpainting...")

    for seed in tqdm.tqdm(seeds):

        rnd = StackedRandomGenerator(device, [seed])

        noise = rnd.randn(
            [1, net.img_channels, H, W],
            device=device
        )

        latents = noise * mask_tensor + x_orig * (1 - mask_tensor)

        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[1], device=device)
            ]
        if class_idx is not None:
            class_labels[:] = 0
            class_labels[:, class_idx] = 1

        sigma_min = 0.002
        sigma_max = 1

        images = latents - net(
            latents,
            t=sigma_max * torch.ones([1,1,1,1], device=device),
            class_labels=class_labels,
            h=(sigma_max-sigma_min)*sigma_max*torch.ones([1,1,1,1], device=device),
            augment_labels=torch.zeros(1,9).to(device)
        )

        images = x_orig * (1 - mask_tensor) + images * mask_tensor

        images_np = (images * 127.5 + 128).clamp(0,255)
        images_np = images_np.to(torch.uint8).permute(0,2,3,1)
        images_np = images_np.cpu().numpy()[0]

        PIL.Image.fromarray(images_np).save(
            os.path.join(outdir, f"{seed:06d}.png")
        )

    print("Done.")
    print(f"Results saved in {outdir}")


if __name__ == "__main__":
    main()