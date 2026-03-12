import torch
import os
import os.path as osp
import sys
import argparse
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
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="compile",
        help="Disable torch.compile.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 mixed precision inference. Use --no-fp16 to disable.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_false",
        dest="fp16",
        help="Disable FP16 mixed precision.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch size for patchwise inference. Use 0 to disable patchwise inference.",
    )
    args = parser.parse_args()

    return args


def patchwise_inference(img, model, patch_size, scale, fp16, device):
    import torch.nn.functional as F

    _, C, h, w = img.size()
    split_token_h = h // patch_size + 1
    split_token_w = w // patch_size + 1

    mod_pad_h = (split_token_h - h % split_token_h) if h % split_token_h != 0 else 0
    mod_pad_w = (split_token_w - w % split_token_w) if w % split_token_w != 0 else 0
    img = F.pad(img, (0, mod_pad_w, 0, mod_pad_h), "reflect")

    _, _, H, W = img.size()
    split_h = H // split_token_h
    split_w = W // split_token_w
    shave_h = split_h // 10
    shave_w = split_w // 10

    slices = []
    for i in range(split_token_h):
        for j in range(split_token_w):
            top = slice(
                max(0, i * split_h - shave_h), min(H, (i + 1) * split_h + shave_h)
            )
            left = slice(
                max(0, j * split_w - shave_w), min(W, (j + 1) * split_w + shave_w)
            )
            slices.append((top, left))

    outputs = []
    for top, left in tqdm(slices, desc="Processing patches", unit="patch"):
        patch = img[..., top, left]
        if fp16 and device != "cpu":
            with torch.amp.autocast('cuda'):
                out = model(patch)
        else:
            out = model(patch)
        outputs.append(out)

    result = torch.zeros(1, C, H * scale, W * scale, device=img.device)
    for i in range(split_token_h):
        for j in range(split_token_w):
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
            result[..., top, left] = outputs[i * split_token_w + j][..., _top, _left]

    result = result[
        :, :, 0 : h * scale - mod_pad_h * scale, 0 : w * scale - mod_pad_w * scale
    ]
    return result


def process_image(
    image_input_path, image_output_path, model, device, scale, patch_size=0, fp16=False
):
    with torch.no_grad():
        image_input = Image.open(image_input_path).convert("RGB")
        image_input = transforms.ToTensor()(image_input).unsqueeze(0).to(device)

        if patch_size > 0:
            image_output = patchwise_inference(
                image_input, model, patch_size, scale, fp16, device
            )
            image_output = torch.nan_to_num(image_output).clamp(0.0, 1.0)[0].cpu()
        else:
            if fp16 and device != "cpu":
                with torch.amp.autocast('cuda'):
                    image_output = torch.nan_to_num(model(image_input)).clamp(0.0, 1.0)[0].cpu()
            else:
                image_output = torch.nan_to_num(model(image_input)).clamp(0.0, 1.0)[0].cpu()

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
                    args.fp16,
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
                args.fp16,
            )


if __name__ == "__main__":
    main()
