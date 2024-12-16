import argparse, os, sys, glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from tqdm import tqdm
from datasets import load_dataset
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMTextToImagePipeline


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
    parser.add_argument(
        "--prompt_data",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="mscoco"
    )
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
        "--scheduler",
        type=str,
        default="ddim",
        choices=["ddim", "dpm"],
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
        default=5,
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
        "--class_end_idx", type=int, help="class number end_idx", default=100
    )


    opt = parser.parse_args()

    # get diffusers pipeline
    # load pipeline
    pipeline = LDMTextToImagePipeline.from_pretrained(opt.pretrained_model_name_or_path)    
    # replace a faster scheduler
    if opt.scheduler == 'ddim':
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif opt.scheduler == 'dpm':
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to('cuda')
    pipeline = pipeline.to(torch.float16)
    
    if opt.prompt_data == "mscoco":
        dataset = load_dataset("HuggingFaceM4/COCO", name="2014_captions", split="validation", num_proc=4)
        # dataset = load_dataset("shunk031/MSCOCO", name="2017_captions", split="validation", num_proc=4, cache_dir='/ocean/projects/cis220031p/hchen10/datasets/hf_datasets')
    elif opt.prompt_data == "parti":
        dataset = load_dataset('nateraw/parti-prompts', split="train")
    
    # output dir
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    
    # get class labels
    n_samples_per_prompt = opt.n_samples
    base_count = 0
    batch_size = opt.batch_size
    idx_range = range(opt.class_start_idx, opt.class_end_idx)
    
    with torch.no_grad():
        
        for idx, data in tqdm(enumerate(dataset), total=len(dataset)):
            
            if idx not in idx_range:
                continue
            
            if opt.prompt_data == "mscoco":
                
                image_id = data['imgid']
                prompt_list = data['sentences_raw']
                prompt_id_list = data['sentids']
                
                sample_path = os.path.join(outpath, f"{image_id}_{idx}")
                os.makedirs(sample_path, exist_ok=True)
                
                # wrtie a meta list
                with open(f"{sample_path}/meta.txt", "w") as f:
                    for idx, prompt in enumerate(prompt_list):
                        f.write(f"{prompt_id_list[idx]}: {prompt}\n")
                
                all_prompt_list = [p for p in prompt_list for _ in range(n_samples_per_prompt)]
                all_prompt_id_list = [p for p in prompt_id_list for _ in range(n_samples_per_prompt)]
                        
            elif opt.prompt_data == "parti":
                
                prompt = data['Prompt']
                catagory = data['Category']
                challenge = data['Challenge']
                note = data['Note']
                
                sample_path = os.path.join(outpath, f"{idx}_{catagory}_{challenge}")
                os.makedirs(sample_path, exist_ok=True)
                base_count = 0
                
                with open(f"{sample_path}/meta.txt", "w") as f:
                    f.write(f"Prompt: {prompt}\n")
                    f.write(f"Catagory: {catagory}\n")
                    f.write(f"challenge: {challenge}\n")
                    f.write(f"note: {note}\n")
                
                prompt_list = [prompt]
                all_prompt_list = [prompt for _ in range(n_samples_per_prompt)]

            print(all_prompt_list)
            print(f"rendering {n_samples_per_prompt} examples for {len(prompt_list)} prompts in {opt.ddim_steps} steps and using s={opt.scale:.2f}.")
            base_count = 0
            
            for idx in range(0, len(all_prompt_list), batch_size):
                prompt_list = all_prompt_list[idx:idx+batch_size]
                
                start_code = None
                if opt.fixed_code:
                    start_code = torch.randn([len(prompt_list), 3, 64, 64], device=torch.device('cuda')).to(torch.float16)

                images = pipeline(prompt_list, num_inference_steps=opt.ddim_steps, eta=opt.ddim_eta, guidance_scale=opt.scale, latents=start_code).images
                
                if opt.prompt_data == "mscoco":
                    prompt_id_list = all_prompt_id_list[idx:idx+batch_size]
                    save_names = []
                    for prompt_id, image_idx in zip(prompt_id_list, range(len(images))):
                        save_name = f"{sample_path}/{prompt_id}_{base_count+image_idx:05}.jpeg"
                        save_names.append(save_name)
                        
                elif opt.prompt_data == "parti":
                    save_names = [f"{sample_path}/{base_count+idx:05}.jpeg" for idx in range(len(images))]
                base_count += len(images)
                
                for image, save_name in zip(images, save_names):
                    save_image(image, save_name)
    
    
if __name__ == '__main__':
    main()