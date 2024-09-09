import os, sys, shutil
import json
import toml
from pathlib import Path
import argparse
import copy
import datetime

def augment_parse():
    parser = argparse.ArgumentParser(description='Augment data for training')
    parser.add_argument('--work_dir', type=str, 
                        default=None, required=True,
                        help='Directory to work')
    parser.add_argument('--subfolders', action='store_true',
                        help='Process subfolders')
    parser.add_argument('--init_new', action='store_true',
                        help='Initialize new data')
    parser.add_argument('--pretrained_model_name_or_path', type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help='Pretrained model name or path')
    parser.add_argument('--add_vae', action='store_true',
                        help='addtional vae')
    parser.add_argument('--batch_size', type=int,
                        default=5,
                        help='Batch size')
    parser.add_argument('--rank', type=int,
                        default=128,
                        help='Rank')
    parser.add_argument('--alpha', type=float,
                        default=1.0,
                        help='Alpha')
    parser.add_argument('--max_train_steps', type=int,
                        default=1000,
                        help='Max train steps')
    parser.add_argument('--prodigy', action='store_true',
                        help='Use prodigy optimizer')
    return parser.parse_args()
    
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
    

def train_process(source_dir, args):
    dataset_config_path = source_dir / 'config' / 'data_set.toml'
    cmd_out_config_file = source_dir / 'config' / 'train_out.toml'
    output_dir = source_dir / 'models'
    if args.init_new and output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = source_dir / 'logs'
    if args.init_new and log_dir.exists():
        shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    wandb_name = source_dir.name
    
    model_files = output_dir.glob('*.safetensors')
    donefile = output_dir / 'done.txt'
    if donefile.exists() and len(list(model_files)) > 0:
        print(f"Skip {source_dir.name}")
        return 0
    cmdfile = output_dir / 'cmd.txt'
    
    base_model = args.pretrained_model_name_or_path
    # base_model = "/home/wangyh/sdxl_models/checkpoint/juggernautXL_v9Rundiffusionphoto2.safetensors"
    starttime = datetime.datetime.now()
    cmd = f'accelerate launch --num_cpu_threads_per_process 4 sd-scripts/sdxl_train_network.py'
    cmd += f' --pretrained_model_name_or_path="{base_model}"'
    if args.add_vae:
        cmd += f' --vae="/home/wangyh/sdxl_models/vae/madebyollin-sdxl-vae-fp16-fix.safetensors"'
    cmd += f' --dataset_config="{str(dataset_config_path)}"'
    cmd += f' --output_dir="{str(output_dir)}"'
    cmd += f' --logging_dir="{str(log_dir)}"'
    cmd += f' --train_batch_size={args.batch_size}'
    cmd += f' --resolution="1024"'
    cmd += f' --seed=789987'
    cmd += f' --output_name="lora_weights"'
    cmd += f' --save_model_as=safetensors'
    cmd += f' --max_train_steps={args.max_train_steps}'
    cmd += f' --save_every_n_steps=100'
    # cmd += f' --save_n_epoch_ratio=10'
    if args.prodigy:
        cmd += f' --optimizer_type="Prodigy"'
        cmd += f' --learning_rate=1.0'
        cmd += f' --unet_lr=1.0 --text_encoder_lr=1.0'
        cmd += f' --optimizer_args safeguard_warmup=True weight_decay=0.01 betas=0.9,0.99 use_bias_correction=True decouple=True d_coef=2'
        cmd += f' --lr_scheduler="cosine"'
        cmd += f' --lr_warmup=0'
    else:
        cmd += f' --optimizer_type="AdamW"'
        cmd += f' --learning_rate=1e-4'
        cmd += f' --unet_lr=1e-4 --text_encoder_lr=1e-5'
        cmd += f' --optimizer_args weight_decay=0.01 betas=0.9,0.99'
        cmd += f' --lr_scheduler="cosine"'
        cmd += f' --lr_warmup=0'
    cmd += f' --min_snr_gamma=5.0'
    cmd += f' --xformers'
    cmd += f' --cache_latents'
    cmd += f' --gradient_checkpointing'
    cmd += f' --mixed_precision="bf16"'
    cmd += f' --save_precision="fp16"'
    cmd += f' --network_train_unet_only'
    cmd += f' --network_module=networks.lora'
    cmd += f' --network_args conv_dim=64 conv_alpha=1.0'
    # cmd += f' --network_module=lycoris.kohya'
    # cmd += f' --network_args preset=full algo=lora conv_dim=64 conv_alpha=1.0 train_norm=True'
    # cmd += f' --network_args preset=attn-mlp algo=lora'
    cmd += f' --network_dim={args.rank}'
    cmd += f' --network_alpha={args.alpha}'
    cmd += f' --log_with=wandb --wandb_run_name="{wandb_name}"'
    cmd += f' --wandb_api_key="763864d93043b06fb3556826407de609937819b1"'
    cmd += f' --prior_loss_weight=1.0'
    # cmd += f' --save_state_on_train_end'

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

def main(args):
    work_dir = Path(args.work_dir)
    dir_list = []
    if args.subfolders:
        for folder in work_dir.iterdir():
            if folder.is_dir():
                dir_list.append(folder)
    else:
        dir_list.append(work_dir)
    dir_list = sorted(dir_list, key=lambda x: x.name)
    for folder in dir_list:
        ret = train_process(folder, args)
        if ret != 0:
            print(f"Error in {folder.name}")
            return ret
    return 0

if __name__ == '__main__':
    args = augment_parse()
    ret = main(args)
    sys.exit(ret)