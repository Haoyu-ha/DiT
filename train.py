import os
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy
from glob import glob
from time import time
import argparse
from models.DiT import DiT_models
from diffusion.gaussian_diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils.training_utils import update_ema, requires_grad, create_logger


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    args.device = torch.device("cuda")
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(args.device)
    ema = deepcopy(model).to(args.device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    diffusion = create_diffusion(diffusion_steps=1000)  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(args.device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup dataset and data loader:
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 将图像大小调整为32x32
        transforms.Lambda(lambda x: x.convert("RGB")),  # 将灰度图像转换为RGB
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对于三个通道的归一化
    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Start training
    train(args, model, ema, diffusion, vae, loader, optimizer, logger, checkpoint_dir)


def train(args, model, ema, diffusion, vae, loader, optimizer, logger, checkpoint_dir):
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x, y = x.to(args.device), y.to(args.device)

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=args.device)
            noise = torch.randn_like(x)
            x_t = diffusion.q_sample(x, t, noise)

            model_output = model(x_t, t, y)

            loss_dict = diffusion.training_losses(model_output, x, x_t, t, noise)
            loss = loss_dict["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": optimizer.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        # save each epoch
        # checkpoint = {
        #     "model": model.state_dict(),
        #     "ema": ema.state_dict(),
        #     "opt": optimizer.state_dict(),
        #     "args": args
        # }
        # checkpoint_path = f"./ckpt/newest.pt"
        # torch.save(checkpoint, checkpoint_path)

    logger.info(f"Saved checkpoint to {checkpoint_path}")
    model.eval()  # important! This disables randomized embedding dropout
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/MNIST/raw")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    args = parser.parse_args()
    main(args)
