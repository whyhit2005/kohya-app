import os, sys, shutil
import argparse
from pathlib import Path
import uuid

def augment_parse():
    parser = argparse.ArgumentParser(description='Augment data for training')
    parser.add_argument('--work_dir', type=str, 
                        default=None, required=True,
                        help='Directory to work')
    parser.add_argument('--ckpt', type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0", 
                        help='Checkpoint file')
    parser.add_argument('--add_vae', action='store_true',
                        help='Add VAE')
    parser.add_argument('--prompt', type=str,
                        default="solo, look at viewer, woman armor, sexy pose, full body, face focus, lake, spring",
                        help='Prompt')
    parser.add_argument('--from_file', type=str,
                        default=None,
                        help='Prompt from file')
    parser.add_argument('--sample_num', type=int,
                        default=20,
                        help='Number of samples')
    parser.add_argument('--subfolders', action='store_true',
                        help='Process subfolders')
    parser.add_argument('--init_new', action='store_true',
                        help='Initialize new data')
    parser.add_argument('--checkpoint', action='store_true',
                        help='Use checkpoint')
    parser.add_argument('--checkpoint_range', type=str,
                        default=None,
                        help='test checkpoint range')
    parser.add_argument('--sample_dir', type=str,
                        default="samples",
                        help='Sample directory')
    parser.add_argument('--sub_range', type=str,
                        default=None,
                        help='Start at')
    return parser.parse_args()

def infer(source_dir, lora_file, args):
    image_dir = source_dir / 'images'
    sample_dir = source_dir / args.sample_dir
    sample_dir.mkdir(parents=True, exist_ok=True)
    temp_name = str(uuid.uuid4())[:8]
    temp_dir = source_dir / "temp" / temp_name
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    caption_files = list(image_dir.glob('*.txt'))
    with open(caption_files[0], 'r') as f:
        captions = f.readlines()
        tokens = captions[0].strip().split(", ")
        instance_prompt = tokens[0]
    prompt = f"photo of {instance_prompt}"
    prompt += f", {args.prompt}"
    prompt += f", masterpeice, perfect, high quality, ultra detail"
    
    cmd = f"python sd-scripts/sdxl_gen_img.py"
    cmd += f" --ckpt /home/wangyh/sdxl_models/checkpoint/sd_xl_base_1.0.safetensors"
    if args.add_vae:
        cmd += f" --vae /home/wangyh/sdxl_models/vae/madebyollin-sdxl-vae-fp16-fix.safetensors"
    cmd += f" --outdir {str(temp_dir)}"
    cmd += f" --xformers --fp16"
    cmd += f" --W 1024 --H 1024"
    cmd += f" --steps 50 --scale 10.0 --seed 54321"
    cmd += f" --images_per_prompt {args.sample_num}"
    if args.from_file:
        cmd += f" --from_file \"{args.from_file}\""
    else:
        cmd += f" --prompt \"{prompt}\""
    cmd += f" --sequential_file_name"
    cmd += f" --sampler \"dpmsolver++\""
    cmd += f" --network_module networks.lora"
    cmd += f" --network_weights \"{str(lora_file)}\""
    cmd += f" --network_mul 1.0"
    cmd += f" --network_merge_n_models 1"
    cmd += f" --network_merge"
    ret = os.system(cmd)
    if ret != 0:
        return ret
    
    suffix = "final"
    if args.checkpoint:
        start = lora_file.stem.find("step")
        if start != -1:
            suffix_i = int(lora_file.stem[start+4:])
            suffix = f'{suffix_i:04d}'
    for tfile in temp_dir.iterdir():
        if tfile.suffix in [".jpg", ".png", ".jpeg"]:
            shutil.move(tfile, sample_dir / f"{tfile.stem}_{suffix}{tfile.suffix}")
    shutil.rmtree(temp_dir, ignore_errors=True)
    return ret

def main(args):
    work_dir = Path(args.work_dir)
    dirlist = []
    if args.subfolders:
        for dir in work_dir.iterdir():
            if dir.is_dir():
                dirlist.append(dir)
    else:
        dirlist.append(work_dir)
    dirlist = sorted(dirlist, key=lambda x: x.name)
    if args.sub_range is not None:
        ranges = args.sub_range.split(",")
        start, end = int(ranges[0]), int(ranges[1])
        start = max(0, start)
        end = min(len(dirlist), end)
        dirlist = dirlist[start:end]
    
    inputlist = []
    for tdir in dirlist:
        mdir = tdir / "models"
        if args.checkpoint:
            model_files = list(mdir.glob("*step*.safetensors"))
            model_files = sorted(model_files, key=lambda x: x.name)
            if args.checkpoint_range:
                crange = [int(ci) for ci in args.checkpoint_range.split(",")]
                crange[0] = max(0, crange[0])
                crange[1] = min(len(model_files), crange[1])
                model_files = model_files[crange[0]:crange[1]]
        else:
            model_files = [mdir / "lora_weights.safetensors"]
        model_files = sorted(model_files, key=lambda x: x.name, reverse=True)
        inputlist.append((tdir, model_files))
    
    for tdir, lora_files in inputlist:
        sdir = tdir / args.sample_dir
        if args.init_new and sdir.exists():
            shutil.rmtree(sdir, ignore_errors=True)
        sdir.mkdir(parents=True, exist_ok=True)
        for lora_file in lora_files:
            ret = infer(tdir, lora_file, args)
            if ret != 0:
                print(f"Error in {tdir.name}")
                return ret
    return 0

if __name__ == '__main__':
    args = augment_parse()
    sys.exit(main(args))