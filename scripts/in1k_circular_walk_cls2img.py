import argparse, os, sys, glob
import yaml 
import torch
import numpy as np
import math
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from torchvision.utils import make_grid
from diffusers import DPMSolverMultistepScheduler
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import ClassEmbedder, LDMClassToImagePipeline
from diffusers import AutoencoderKL, DDIMScheduler, DiTPipeline, Transformer2DModel, DPMSolverMultistepScheduler


def save_image(array_img, save_path):
    img = Image.fromarray(array_img)
    img.save(save_path, "JPEG", quality=100, subsampling=0)


def save_images_multiprocess(mp_save_count, gen_images_list, save_names_list):
    import multiprocessing
    pool = multiprocessing.Pool(mp_save_count)
    pool.starmap(save_image, zip(gen_images_list, save_names_list))
    pool.close()
    pool.join()


def load_ldm_model(model_name):
    pipeline = LDMClassToImagePipeline.from_pretrained(model_name)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler .config)    
    pipeline.unet.to(memory_format=torch.channels_last)
    pipeline.vqvae.to(memory_format=torch.channels_last)
    return pipeline


def load_dit_model(model_name):
    pipeline = DiTPipeline.from_pretrained(model_name)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler .config)    
    return pipeline
    


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_list",
        nargs="+",
        default=['DiffusionNoise/ldm_imagenet_clean', 'DiffusionNoise/ldm_imagenet_random_noise_2.5', 'DiffusionNoise/ldm_imagenet_random_noise_5',  'DiffusionNoise/ldm_imagenet_random_noise_10'],
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="experiments/ldm_imagenet_vis"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=25,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help='image size to genearte'
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--walk_steps",
        type=int,
        default=20,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    
    # split classes
    parser.add_argument(
        "--class_synset", type=str, help="class synset", default=0, nargs='+',
    )
    
    opt = parser.parse_args()

    if opt.seed is None:
        generator = None
    else:
        generator = torch.Generator(device='cuda').manual_seed(opt.seed)   

    model_list = opt.model_list
    if isinstance(model_list, str):
        model_list = [model_list]


    # output dir
    opt.outdir = os.path.join(opt.outdir, f'models{len(model_list)}_dpm_steps{opt.ddim_steps}_scale{opt.scale}_seed{opt.seed}')
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # get class labels
    # classes = np.arange(opt.class_start_idx, opt.class_end_idx)
    class_synset = opt.class_synset
    if isinstance(class_synset, str):
        class_synset = [class_synset]
    n_samples_per_class = opt.n_samples
    batch_size = opt.batch_size

    with open('data/synset_human.txt', "r") as f:
        syn2name_dict = f.read().splitlines()
        syn2name_dict = dict(line.split(maxsplit=1) for line in syn2name_dict)
    with open('data/index_synset.yaml', "r") as f:
        idx2syn_dict = yaml.safe_load(f)
    syn2idx_dict = {v:k for k,v in idx2syn_dict.items()}
    
    # load pipelines
    model_pipeline_dict = {}
    for model in model_list:
        if 'ldm' in model:
            model_pipeline_dict[model] = load_ldm_model(model)
        elif 'dit' in model:
            model_pipeline_dict[model] = load_dit_model(model)
        else:
            raise ValueError(f"Unknown model type: {model}")
    
    for synset in class_synset:
        class_label = syn2idx_dict[synset]
        folder_name = idx2syn_dict[class_label]
        class_human_name = syn2name_dict[folder_name].split(',')[0]

        
        for i in tqdm(range(n_samples_per_class)):
            
            sample_path = os.path.join(outpath, folder_name, f'walk_{i:03d}')
            os.makedirs(sample_path, exist_ok=True)
            
            # get circular walk latents
            walk_noise_x = torch.randn([3, 64, 64], generator=generator, device='cuda')
            walk_noise_y = torch.randn([3, 64, 64], generator=generator, device='cuda')
            
            walk_scale_x = torch.cos(torch.linspace(0, 2, opt.walk_steps) * math.pi).to('cuda')
            walk_scale_y = torch.sin(torch.linspace(0, 2, opt.walk_steps) * math.pi).to('cuda')
            noise_x = torch.tensordot(walk_scale_x, walk_noise_x, dims=0)
            noise_y = torch.tensordot(walk_scale_y, walk_noise_y, dims=0)
            latents = torch.add(noise_x, noise_y).to(torch.float16)
                
            all_images = []
            for model in model_list:
                
                model_images = []
                pipeline = model_pipeline_dict[model].to('cuda')
                pipeline = pipeline.to(torch.float16)
                model_generator = torch.Generator(device='cuda').manual_seed(int(opt.seed) + int(class_label))
                
                for j in range(0, len(latents), batch_size):
                    
                    if 'ldm' in model:
                        images = pipeline([class_label] * batch_size, num_inference_steps=opt.ddim_steps, eta=opt.ddim_eta, guidance_scale=opt.scale, latents=latents[j:j+batch_size], generator=model_generator, output_type='np').images
                    elif 'dit' in model:
                        raise NotImplementedError("DIT model not implemented")
                    images = (images * 255.0).astype("uint8")
                    model_images.append(images)
                
                model_name = model.split('/')[-1]
                model_images = np.concatenate(model_images, axis=0)
                os.makedirs(os.path.join(sample_path, model_name), exist_ok=True)
                save_path_list = [os.path.join(sample_path, f"{model_name}/{class_human_name}_{i:03d}.jpeg") for i in range(len(model_images))]
                save_images_multiprocess(4, model_images, save_path_list)
                
                # save individual model images
                model_images = torch.from_numpy(model_images).permute(0, 3, 1, 2)
                grid_image = make_grid(model_images, nrow=len(model_images), padding=0, pad_value=255)
                all_images.append(grid_image.unsqueeze(0))
            
            # save grid images for all models
            all_images = torch.cat(all_images, dim=0)
            grid_image = make_grid(all_images, nrow=1, padding=2, pad_value=255)
            grid_image = grid_image.permute(1, 2, 0).numpy()
            save_path = os.path.join(sample_path, f"{class_human_name}_grid.jpeg")
            Image.fromarray(grid_image).save(save_path)
    
    
if __name__ == '__main__':
    main()