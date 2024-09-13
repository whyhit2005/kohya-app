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
    
def infer(cur_dir, sample_dir, models, cfg_args):
    prompt = None
    prompt_file = None
    if cfg_args.prompt_file:
        prompt_file = rewrite_prompt(cur_dir, cfg_args)
    elif cfg_args.prompt:
        prompt = cfg_args.prompt
    else:
        raise ValueError("prompt or prompt_file must be provided")
    
    output_dir = sample_dir
    base_model, vae_model = models
    cmd = f"python sd-scripts/sdxl_gen_img.py \\\n"
    cmd += f" --ckpt {str(base_model)} \\\n"
    if vae_model:
        cmd += f" --vae {vae_model} \\\n"
    cmd += f" --outdir {str(output_dir)} \\\n"
    cmd += f" --xformers --fp16 \\\n"
    cmd += f" --W 1024 --H 1024 \\\n"
    cmd += f" --steps {cfg_args.infer_steps} \\\n"
    cmd += f" --scale {cfg_args.scale} --seed {cfg_args.seed} \\\n"
    cmd += f" --images_per_prompt {cfg_args.sample_num} \\\n"
    if prompt_file:
        cmd += f" --from_file \"{prompt_file}\" \\\n"
    else:
        cmd += f" --prompt \"{prompt}\" \\\n"
    cmd += f" --sequential_file_name"
    cmd += f" --sampler euler_a"
    ret = os.system(cmd)
    return ret


def main(cfg_args):
    work_dir = Path(cfg_args.output_dir)
    base_models = cfg_args.pretrained_model_name_or_path
    vae_models = cfg_args.pretrained_vae_model_name_or_path
    models = list(zip(base_models, vae_models))
    for cur_models in models:
        base_model, vae_model = cur_models
        base_model = Path(base_model)
        task_name = base_model.stem
        sample_path = work_dir / task_name
        if cfg_args.init_new and sample_path.exists():
            shutil.rmtree(sample_path, ignore_errors=True)
        sample_path.mkdir(parents=True, exist_ok=True)
        ret = infer(work_dir, sample_path, cur_models, cfg_args)
        if ret != 0:
            logging.error(f"Error in {task_name}")
            return ret
    return 0

if __name__ == '__main__':
    args = parse_args()
    cfg_args = load_config_yaml(args)
    ret = main(cfg_args)
    sys.exit(ret)