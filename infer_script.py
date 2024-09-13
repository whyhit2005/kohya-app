import os, sys, shutil
import argparse
from pathlib import Path
import uuid
from omegaconf import OmegaConf
import yaml
import logging
logging.basicConfig(level=logging.INFO)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="kohya generate images")
    parser.add_argument("cfg_file", type=str,
                        help="config file")
    return parser.parse_args(input_args)


def load_config_yaml(args):
    with open(args.cfg_file, "r") as f:
        cfg_data = yaml.safe_load(f)
        t_data = {}
        t_data.update(cfg_data["base"])
        t_data.update(cfg_data["infer"])
        t_data["train"] = cfg_data["train"]
        cfg_args = OmegaConf.create(t_data)
        return cfg_args

   
def rewrite_prompt(cur_dir, cfg_args):
    source_file = Path(cfg_args.prompt_file)
    with open(source_file, 'r') as f:
        prompts = f.readlines()
    output_dir = cur_dir / 'prompts'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / source_file.name
    with open(output_file, 'w') as f:
        concept = f'{cfg_args.instance_token} {cfg_args.class_name}'
        for prompt in prompts:
            if cfg_args.replace_word:
                prompt = prompt.replace(cfg_args.replace_word, concept)
            if cfg_args.prompt_prefix:
                prompt = f'{cfg_args.prompt_prefix}, {prompt}'
            f.write(prompt)
    return output_file

 
def infer(cur_dir, sample_dir, lora_file, cfg_args):
    pormpt_file = None
    if cfg_args.prompt_file:
        prompt_file = rewrite_prompt(cur_dir, cfg_args)
    
    output_dir = sample_dir
    if "step" in lora_file.name:
        output_dir = sample_dir / lora_file.stem.split("-")[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_model = cfg_args.pretrained_model_name_or_path
    vae_model = cfg_args.pretrained_vae_model_name_or_path
    cmd = f"python sd-scripts/sdxl_gen_img.py \\\n"
    cmd += f" --ckpt {base_model} \\\n"
    if vae_model:
        cmd += f" --vae {vae_model} \\\n"
    cmd += f" --outdir {str(output_dir)} \\\n"
    cmd += f" --xformers --fp16 \\\n"
    cmd += f" --W 1024 --H 1024 \\\n"
    cmd += f" --steps {cfg_args.infer_steps} \\\n"
    cmd += f" --scale 7.0 --seed 666666 \\\n"
    cmd += f" --images_per_prompt {cfg_args.sample_num} \\\n"
    if prompt_file:
        cmd += f" --from_file \"{prompt_file}\" \\\n"
    else:
        cmd += f" --prompt \"{cfg_args.prompt}\" \\\n"
    cmd += f" --sequential_file_name \\\n"
    cmd += f" --network_module networks.lora \\\n"
    # if cfg_args.train.conv_rank is not None:
    #     cmd += f' --network_args conv_dim={cfg_args.train.conv_rank} conv_alpha={cfg_args.train.conv_rank_alpha} \\\n'
    cmd += f" --network_weights \"{str(lora_file)}\" \\\n"
    cmd += f" --network_mul 1.0 \\\n"
    # cmd += f" --network_merge_n_models 1"
    # cmd += f" --network_merge"
    ret = os.system(cmd)
    return ret


def main(cfg_args):
    work_dir = Path(cfg_args.work_dir)
    dirlist = []
    if cfg_args.subfolders:
        for dir in work_dir.iterdir():
            if dir.is_dir():
                dirlist.append(dir)
    else:
        dirlist.append(work_dir)
    dirlist = sorted(dirlist, key=lambda x: x.name)
    if cfg_args.sub_range is not None:
        ranges = cfg_args.sub_range.split(",")
        start, end = int(ranges[0]), int(ranges[1])
        start = max(0, start)
        end = min(len(dirlist), end)
        dirlist = dirlist[start:end]
    
    inputlist = []
    for tdir in dirlist:
        mdir = tdir / cfg_args.model_dir_name / cfg_args.task_name
        if cfg_args.checkpoint:
            model_files = list(mdir.glob("*step*.safetensors"))
            model_files = sorted(model_files, key=lambda x: x.name)
            crange = [int(ci) for ci in cfg_args.checkpoint.split(",")]
            crange[0] = max(0, crange[0])
            crange[1] = min(len(model_files), crange[1])
            model_files = model_files[crange[0]:crange[1]]
        else:
            model_files = [mdir / "lora_weights.safetensors"]
        model_files = sorted(model_files, key=lambda x: x.name, reverse=True)
        inputlist.append((tdir, model_files))
    
    for tdir, lora_files in inputlist:
        sdir = tdir / cfg_args.sample_dir_name /cfg_args.task_name / cfg_args.sample_task_name
        if cfg_args.init_new and sdir.exists():
            shutil.rmtree(sdir, ignore_errors=True)
        sdir.mkdir(parents=True, exist_ok=True)
        for lora_file in lora_files:
            ret = infer(tdir, sdir, lora_file, cfg_args)
            if ret != 0:
                logging.error(f"Error in {tdir.name}")
                return ret
    return 0


if __name__ == '__main__':
    args = parse_args()
    cfg_args = load_config_yaml(args)
    ret = main(cfg_args)
    sys.exit(ret)