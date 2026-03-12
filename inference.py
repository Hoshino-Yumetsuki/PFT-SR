import torch
import os
import os.path as osp
import argparse
import torch.nn.functional as F
from tqdm import tqdm

from PIL import Image
from torchvision import transforms
from basicsr.archs.pft_arch import PFT

model_path = {
    "classical": {
        "2": "experiments/pretrained_models/001_PFT_SRx2_scratch.pth",
        "3": "experiments/pretrained_models/002_PFT_SRx3_finetune.pth",
        "4": "experiments/pretrained_models/003_PFT_SRx4_finetune.pth",
    },
    "lightweight": {
        "2": "experiments/pretrained_models/101_PFT_light_SRx2_scratch.pth",
        "3": "experiments/pretrained_models/102_PFT_light_SRx3_finetune.pth",
        "4": "experiments/pretrained_models/103_PFT_light_SRx4_finetune.pth",
    },
}


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-i", "--in_path", type=str, default="", help="Input image or directory path."
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="results/test/",
        help="Output directory path.",
    )
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument(
        "--task",
        type=str,
        default="classical",
        choices=["classical", "lightweight"],
        help="Task for the model. classical: for classical SR models. lightweight: for lightweight models.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Enable torch.compile (experimental, may not work with custom ops).",
    )
    # parser.add_argument(
    #     "--no-compile",
    #     action="store_false",
    #     dest="compile",
    #     help="Disable torch.compile.",
    # )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch size for patchwise inference. Larger = fewer patches = faster overall (less kernel launch overhead). "
        "Use 0 to disable. Rule of thumb for 8 GB VRAM: 256 (safe), 384 (faster), 512 (may OOM on dense-attention layers).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of patches to process in parallel. Higher = faster but uses MUCH more VRAM. Default 1 is safest for 8 GB GPUs.",
    )
    args = parser.parse_args()

    return args


def patchwise_inference(img, model, patch_size, scale, batch_size=4):
    _, C, h, w = img.size()

    # Number of tiles in each dimension
    split_token_h = max(1, (h + patch_size - 1) // patch_size)
    split_token_w = max(1, (w + patch_size - 1) // patch_size)

    # Pad image so it divides evenly into tiles
    mod_pad_h = (split_token_h - h % split_token_h) % split_token_h
    mod_pad_w = (split_token_w - w % split_token_w) % split_token_w
    img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), mode="reflect")

    _, _, H, W = img.size()
    split_h = H // split_token_h
    split_w = W // split_token_w
    # Overlap: fixed 8 px minimum, or 10% of tile size
    shave_h = max(8, split_h // 10)
    shave_w = max(8, split_w // 10)

    # Canonical (max) patch dimensions for uniform batching
    ph_max = split_h + 2 * shave_h
    pw_max = split_w + 2 * shave_w

    # Collect raw slices and actual patch H/W (edge tiles may be smaller)
    patches_meta = []  # (top_slice, left_slice, actual_ph, actual_pw)
    for i in range(split_token_h):
        for j in range(split_token_w):
            t = max(0, i * split_h - shave_h)
            b = min(H, (i + 1) * split_h + shave_h)
            left_ = max(0, j * split_w - shave_w)
            r = min(W, (j + 1) * split_w + shave_w)
            patches_meta.append((slice(t, b), slice(left_, r), b - t, r - left_))

    # Extract and pad all patches to uniform size (ph_max, pw_max)
    all_patches = []
    for top, left, ph, pw in patches_meta:
        p = img[..., top, left]  # (1, C, ph, pw)
        if ph < ph_max or pw < pw_max:
            p = F.pad(p, (0, pw_max - pw, 0, ph_max - ph), mode="reflect")
        all_patches.append(p)

    # Batch forward: process batch_size patches at a time
    n_total = len(all_patches)
    outputs = []  # each element: (1, C, ph*scale, pw*scale) — already cropped
    if batch_size == 1:
        # Fast path: avoid the cat/split overhead for single-patch processing
        for i, (p, (_, _, ph, pw)) in enumerate(
            tqdm(
                zip(all_patches, patches_meta),
                total=n_total,
                desc="Processing patches",
                unit="patch",
            )
        ):
            out = model(p)  # (1, C, ph_max*scale, pw_max*scale)
            outputs.append(out[:, :, : ph * scale, : pw * scale])
    else:
        for start in tqdm(
            range(0, n_total, batch_size), desc="Processing patch batches", unit="batch"
        ):
            chunk = all_patches[
                start : start + batch_size
            ]  # list of (1, C, ph_max, pw_max)
            batch = torch.cat(chunk, dim=0)  # (B, C, ph_max, pw_max)
            out = model(batch)  # (B, C, ph_max*scale, pw_max*scale)
            # Split back and crop to actual output size
            for k, (_, _, ph, pw) in enumerate(
                patches_meta[start : start + batch_size]
            ):
                outputs.append(out[k : k + 1, :, : ph * scale, : pw * scale])

    # Assemble result (match dtype of model outputs for efficiency)
    result = torch.zeros(
        1, C, H * scale, W * scale, device=img.device, dtype=outputs[0].dtype
    )
    for idx, (i, j) in enumerate(
        (i, j) for i in range(split_token_h) for j in range(split_token_w)
    ):
        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
        _top = slice(
            shave_h * scale if i > 0 else 0,
            (shave_h + split_h) * scale if i > 0 else split_h * scale,
        )
        _left = slice(
            shave_w * scale if j > 0 else 0,
            (shave_w + split_w) * scale if j > 0 else split_w * scale,
        )
        result[..., top, left] = outputs[idx][..., _top, _left]

    # Remove padding
    result = result[
        :, :, : h * scale - mod_pad_h * scale, : w * scale - mod_pad_w * scale
    ]
    return result


def process_image(
    image_input_path,
    image_output_path,
    model,
    device,
    scale,
    patch_size=0,
    batch_size=4,
):
    with torch.no_grad():
        image_input = Image.open(image_input_path).convert("RGB")
        image_input = transforms.ToTensor()(image_input).unsqueeze(0).to(device)

        # Match input dtype to model dtype (fp16/fp32)
        model_dtype = next(model.parameters()).dtype
        image_input = image_input.to(dtype=model_dtype)

        if patch_size > 0:
            image_output = patchwise_inference(
                image_input, model, patch_size, scale, batch_size
            )
        else:
            image_output = model(image_input)

        # Debug: Check for NaN/Inf before processing
        if torch.isnan(image_output).any() or torch.isinf(image_output).any():
            print(f"WARNING: Output contains NaN or Inf values!")
            print(f"  NaN count: {torch.isnan(image_output).sum().item()}")
            print(f"  Inf count: {torch.isinf(image_output).sum().item()}")
            print(
                f"  Min: {image_output[~torch.isnan(image_output) & ~torch.isinf(image_output)].min().item() if (~torch.isnan(image_output) & ~torch.isinf(image_output)).any() else 'N/A'}"
            )
            print(
                f"  Max: {image_output[~torch.isnan(image_output) & ~torch.isinf(image_output)].max().item() if (~torch.isnan(image_output) & ~torch.isinf(image_output)).any() else 'N/A'}"
            )
        else:
            print(
                f"Output stats - Min: {image_output.min().item():.6f}, Max: {image_output.max().item():.6f}, Mean: {image_output.mean().item():.6f}"
            )

        image_output = image_output.clamp(0.0, 1.0)[0].cpu().float()
        image_output = transforms.ToPILImage()(image_output)
        image_output.save(image_output_path)


def main():
    args = get_parser()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.task == "classical":
        model = PFT(
            upscale=args.scale,
            embed_dim=240,
            depths=[4, 4, 4, 6, 6, 6],
            num_heads=6,
            num_topk=[
                1024,
                1024,
                1024,
                1024,
                256,
                256,
                256,
                256,
                128,
                128,
                128,
                128,
                64,
                64,
                64,
                64,
                64,
                64,
                32,
                32,
                32,
                32,
                32,
                32,
                16,
                16,
                16,
                16,
                16,
                16,
            ],
            window_size=32,
            convffn_kernel_size=7,
            mlp_ratio=2,
            upsampler="pixelshuffle",
            use_checkpoint=False,
        )
    elif args.task == "lightweight":
        model = PFT(
            upscale=args.scale,
            embed_dim=52,
            depths=[2, 4, 6, 6, 6],
            num_heads=4,
            num_topk=[
                1024,
                1024,
                256,
                256,
                256,
                256,
                128,
                128,
                128,
                128,
                128,
                128,
                64,
                64,
                64,
                64,
                64,
                64,
                32,
                32,
                32,
                32,
                32,
                32,
            ],
            window_size=32,
            convffn_kernel_size=7,
            mlp_ratio=1,
            upsampler="pixelshuffledirect",
            use_checkpoint=False,
        )

    state_dict = torch.load(
        model_path[args.task][str(args.scale)], map_location=device
    )["params_ema"]
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="max-autotune")

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if os.path.isdir(args.in_path):
        for file in os.listdir(args.in_path):
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                image_input_path = osp.join(args.in_path, file)
                file_name = osp.splitext(file)
                image_output_path = os.path.join(
                    args.out_path,
                    file_name[0]
                    + "_PFT_"
                    + args.task
                    + "_SRx"
                    + str(args.scale)
                    + file_name[1],
                )
                process_image(
                    image_input_path,
                    image_output_path,
                    model,
                    device,
                    args.scale,
                    args.patch_size,
                    args.batch_size,
                )
    else:
        if (
            args.in_path.endswith(".png")
            or args.in_path.endswith(".jpg")
            or args.in_path.endswith(".jpeg")
        ):
            image_input_path = args.in_path
            file_name = osp.splitext(osp.basename(args.in_path))
            image_output_path = os.path.join(
                args.out_path,
                file_name[0]
                + "_PFT_"
                + args.task
                + "_SRx"
                + str(args.scale)
                + file_name[1],
            )
            process_image(
                image_input_path,
                image_output_path,
                model,
                device,
                args.scale,
                args.patch_size,
                args.batch_size,
            )


if __name__ == "__main__":
    main()
