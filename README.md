# Slight Corruption in Pre-training Data Makes Better Diffusion Models (NeurIPS 2024 Spotlight) 

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2410.01756-b31b1b.svg)](https://arxiv.org/pdf/2405.20494)&nbsp;
[![huggingface models](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/DiffusionNoise)&nbsp;
<!-- [![demo](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/DiffusionNoise)&nbsp; -->

</div>



## Pre-trained Models

We release all of our pre-trained models (clean and corrupted) under the [DiffusionNoise](https://huggingface.co/DiffusionNoise) organization at huggingface. 
These includes LDM-4 class-to-image and text-to-image models and DiT-XL/2 class-to-image models, with corruption ratio ranging from 0 to 20.


## Scripts

We also provided a bunch of scripts that can be used for utlizing/visualizaing these models.


**LDM-4 Class to Image Generation**
```
python scripts/ldm_in1k_cls2img.py --pretrained_model_name_or_path DiffusionNoise/ldm_imagenet_random_noise_2.5 --outdir visualization/in1k_randnoise_2.5 --ddim_steps 50 --scheduler dpm --batch_size 10 --n_samples 10 --scale 2.25 --fixed_code 
```

**LDM-4 Text to Image Generation**
```
python scripts/ldm_cc3m_text2img.py --pretrained_model_name_or_path DiffusionNoise/ldm_cc3m_random_noise_2.5 --prompt_data mscoco --outdir visualization/cc3m_randnoise_2.5 --ddim_steps 50 --scheduler dpm --batch_size 10 --n_samples 10 --scale 2.25 --fixed_code 
```

**DiT-XL/2 Class to Image Generation**
```
python scripts/dit_in1k_cls2img.py --pretrained_model_name_or_path DiffusionNoise/dit-xl-2_imagenet_random_noise_2.5 --prompt_data mscoco --outdir visualization/cc3m_randnoise_2.5 --ddim_steps 50 --scheduler dpm --batch_size 10 --n_samples 10 --scale 2.25 --fixed_code 
```


**Circular Walk over the Latents**
```
python scripts/in1k_circular_walk_cls2img.py
```


## Reference
```
@article{chen2024slight,
  title={Slight Corruption in Pre-training Data Makes Better Diffusion Models},
  author={Chen, Hao and Han, Yujin and Misra, Diganta and Li, Xiang and Hu, Kai and Zou, Difan and Sugiyama, Masashi and Wang, Jindong and Raj, Bhiksha},
  journal={NeurIPS 2024},
  year={2024}
}
```