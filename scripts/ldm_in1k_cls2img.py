import argparse, os, sys, glob
import yaml 
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMClassToImagePipeline


def save_image(img, save_path):
    img.save(save_path, "JPEG", quality=100, subsampling=0)


def save_images_multiprocess(mp_save_count, gen_images_list, save_names_list):
    import multiprocessing
    pool = multiprocessing.Pool(mp_save_count)
    pool.starmap(save_image, zip(gen_images_list, save_names_list))
    pool.close()
    pool.join()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="DiffusionNoise/ldm_imagenet_random_noise_2.5")
    parser.add_argument("--hub_token", type=str, default="")
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="tmp"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "dpm"],
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
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
        default=50,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
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
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    
    # split classes
    parser.add_argument(
        "--class_start_idx", type=int, help="class number start_idx", default=0
    )
    parser.add_argument(
        "--class_end_idx", type=int, help="class number end_idx", default=1000
    )
    
    # push to hub
    parser.add_argument(
        '--push_to_hub',
        action='store_true',
        help='push model to hub'
    )
    
    opt = parser.parse_args()
    
    pipeline = LDMClassToImagePipeline.from_pretrained(opt.pretrained_model_name_or_path)    
    # replace a faster scheduler
    if opt.scheduler == 'ddim':
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif opt.scheduler == 'dpm':
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to('cuda')
    pipeline = pipeline.to(torch.float16)
    
    # output dir
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # get class labels
    classes = np.arange(opt.class_start_idx, opt.class_end_idx)
    n_samples_per_class = opt.n_samples
    base_count = 0
    batch_size = opt.batch_size
    
    with torch.no_grad():
        for class_label in tqdm(classes):
    
            folder_name = str(class_label) 
            sample_path = os.path.join(outpath, folder_name)
            os.makedirs(sample_path, exist_ok=True)
            base_count = 0
            print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {opt.ddim_steps} steps and using s={opt.scale:.2f}.")
            
            for _ in range(0, n_samples_per_class, batch_size):
            
                start_code = None
                if opt.fixed_code:
                    start_code = torch.randn([batch_size, 3, 64, 64], device=torch.device('cuda')).to(torch.float16)
                
                images = pipeline([class_label] * batch_size, num_inference_steps=opt.ddim_steps, eta=opt.ddim_eta, guidance_scale=opt.scale, latents=start_code).images
                
                save_names = [f"{sample_path}/{base_count+idx:05}.jpeg" for idx in range(len(images))]
                base_count += len(images)
                save_images_multiprocess(8, images, save_names)
    
    
if __name__ == '__main__':
    main()