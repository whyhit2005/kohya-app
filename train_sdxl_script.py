import os, sys, shutil
import json
import toml
from pathlib import Path
import argparse
import copy
import datetime
from omegaconf import OmegaConf
import yaml
import logging
logging.basicConfig(level=logging.INFO)

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="prepare data for training")
    parser.add_argument("cfg_file", type=str,
                        help="config file")
    return parser.parse_args(input_args)

def load_config_yaml(args):
    with open(args.cfg_file, "r") as f:
        cfg_data = yaml.safe_load(f)
        t_data = {}
        t_data.update(cfg_data["base"])
        t_data.update(cfg_data["train"])
        cfg_args = OmegaConf.create(t_data)
        return cfg_args
    

def create_train_toml(source_dir, output_dir, args):
    sname = source_dir.name
    instance = sname.split('-')[-1]
    config_path = output_dir / sname / 'config'
    config_path.mkdir(parents=True, exist_ok=True)
    config_file = config_path / 'train.toml'
    example_file = Path("config/lycoris_full_config.toml")
    example_dict = toml.load(open(example_file, "r"))
    
    train_dict = copy.deepcopy(example_dict)
    train_dict['Basics']['pretrained_model_name_or_path'] = args.pretrained_model_name_or_path
    train_dict['Basics']['batch_size'] = args.batch_size
    del train_dict['Basics']['train_dict']
    train_dict['Basics']['dataset_config'] = str(config_path / 'dataset.toml')
    train_dict['Basics']['resolution']=1024
    train_dict['Save']['output_dir'] = str(output_dir / sname / 'models')
    

def train_process(cur_dir, cfg_args):
    dataset_config_path = cur_dir / 'config' / 'data_set.toml'
    cmd_out_config_file = cur_dir / 'config' / 'train_out.toml'
    output_dir = cur_dir / cfg_args.model_dir_name / cfg_args.task_name
    if cfg_args.init_new and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = cur_dir / cfg_args.log_dir_name
    if cfg_args.init_new and log_dir.exists():
        shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    wandb_name = f'kohya_sdxl_train'
    
    model_files = output_dir.glob('*.safetensors')
    donefile = output_dir / 'done.txt'
    if donefile.exists() and len(list(model_files)) > 0:
        print(f"Skip {cur_dir.name}")
        return 0
    cmdfile = output_dir / 'cmd.txt'
    output_config_file = output_dir / 'train.toml'
    
    base_model = cfg_args.pretrained_model_name_or_path
    vae_model = cfg_args.pretrained_vae_model_name_or_path
    starttime = datetime.datetime.now()
    cmd = f'accelerate launch --num_cpu_threads_per_process 4 sd-scripts/sdxl_train_network.py \\\n'
    cmd += f' --pretrained_model_name_or_path="{base_model}" \\\n'
    if vae_model is not None:
        cmd += f' --vae="{vae_model}" \\\n'
    cmd += f' --dataset_config="{str(dataset_config_path)}" \\\n'
    cmd += f' --output_dir="{str(output_dir)}" \\\n'
    # cmd += f' --config_file="{str(output_config_file)}" \\\n'
    # cmd += f' --output_config \\\n'
    cmd += f' --logging_dir="{str(log_dir)}" \\\n'
    cmd += f' --train_batch_size={cfg_args.train_batch_size} \\\n'
    cmd += f' --resolution="1024" \\\n'
    cmd += f' --seed=789987 \\\n'
    if cfg_args.prior_loss_weight is not None:
        cmd += f' --prior_loss_weight={cfg_args.prior_loss_weight} \\\n'
    cmd += f' --output_name="lora_weights" \\\n'
    cmd += f' --save_model_as=safetensors \\\n'
    cmd += f' --max_train_steps={cfg_args.max_train_steps} \\\n'
    cmd += f' --save_every_n_steps=100 \\\n'
    if cfg_args.prodigy:
        cmd += f' --optimizer_type="Prodigy" \\\n'
        cmd += f' --learning_rate=1.0 \\\n'
        cmd += f' --unet_lr=1.0 --text_encoder_lr=1.0 \\\n'
        cmd += f' --optimizer_args safeguard_warmup=True weight_decay=0.01 betas=0.9,0.99 use_bias_correction=True decouple=True d_coef=1.0 \\\n'
        cmd += f' --lr_scheduler="constant" \\\n'
        cmd += f' --lr_warmup_steps=0 \\\n'
    else:
        cmd += f' --optimizer_type="AdamW" \\\n'
        cmd += f' --learning_rate={cfg_args.unet_lr} \\\n'
        cmd += f' --unet_lr={cfg_args.unet_lr} \\\n'
        cmd += f' --text_encoder_lr={cfg_args.text_encoder_lr} \\\n'
        cmd += f' --optimizer_args weight_decay=0.01 betas=0.9,0.99 \\\n'
        cmd += f' --lr_scheduler="cosine" \\\n'
        cmd += f' --lr_warmup_steps={cfg_args.lr_warmup_steps} \\\n'
    cmd += f' --min_snr_gamma=5.0 \\\n'
    cmd += f' --xformers \\\n'
    # cmd += f' --sdpa \\\n'
    cmd += f' --cache_latents \\\n'
    cmd += f' --gradient_checkpointing \\\n'
    cmd += f' --mixed_precision="bf16" \\\n'
    cmd += f' --save_precision="fp16" \\\n'
    cmd += f' --network_train_unet_only \\\n'
    cmd += f' --network_module=networks.lora \\\n'
    cmd += f' --network_args conv_dim={cfg_args.conv_rank} conv_alpha={cfg_args.conv_rank_alpha} \\\n'
    cmd += f' --network_dim={cfg_args.rank} \\\n'
    cmd += f' --network_alpha={cfg_args.rank_alpha} \\\n'
    if not cfg_args.init_new:
        cmd += f' --resume={str(output_dir)} \\\n'
    cmd += f' --log_with=wandb --wandb_run_name="{wandb_name}" \\\n'
    cmd += f' --wandb_api_key="763864d93043b06fb3556826407de609937819b1" \\\n'
    cmd += f' --save_state_on_train_end'

    with open(cmdfile, 'w') as f:
        f.write(f"{cmd}\n")
    ret = os.system(cmd)
    if ret != 0:
        return ret
    endtime = datetime.datetime.now()
    with open(donefile, 'w') as f:
        f.write(f"Start time: {starttime}\n")
        f.write(f"End time: {endtime}\n")
        f.write(f"lasting time: {endtime-starttime}\n")
        f.write(f"{cmd}\n")
    return ret

def main(cfg_args):
    work_dir = Path(cfg_args.work_dir)
    dir_list = []
    if cfg_args.subfolders:
        for folder in work_dir.iterdir():
            if folder.is_dir():
                dir_list.append(folder)
    else:
        dir_list.append(work_dir)
    dir_list = sorted(dir_list, key=lambda x: x.name)
    for folder in dir_list:
        ret = train_process(folder, cfg_args)
        if ret != 0:
            logging.error(f"Error in {folder.name}")
            return ret
    return 0

def infer_after(args):
    cmd = f"python infer_script.py {args.cfg_file}"
    ret = os.system(cmd)
    return ret


if __name__ == '__main__':
    args = parse_args()
    cfg_args = load_config_yaml(args)
    ret = main(cfg_args)
    if ret == 0:
        ret = infer_after(args)
    else:
        logging.error(f"Error in main")
    sys.exit(ret)