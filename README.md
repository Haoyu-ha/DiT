# Scalable Diffusion Models with Transformers (DiT)

A simplified implementation modified from [DiT](https://github.com/facebookresearch/DiT).
> Paper: **[Scalable Diffusion Models with Transformers](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)**

> More instructions can be found in [DiT](https://github.com/facebookresearch/DiT).

## Training
Single GPU:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py
```
Distributed Training:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 train_ddp.py
```


## Sampling
```bash
python sample.py --ckpt <ckpt_path>
```


## Note
Refer to [DiT](https://github.com/facebookresearch/DiT)  for more instructions.
