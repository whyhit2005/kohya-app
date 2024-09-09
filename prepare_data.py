import os, sys
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
import toml
import json
import re
import random
import warnings
import string
import random

def argument_parse():
    parser = argparse.ArgumentParser(description='Prepare data for training')
    parser.add_argument('--source_dir', type=str, 
                        default=None, required=True,
                        help='Directory to store the data')
    parser.add_argument('--subfolders', action='store_true',
                        default=False,
                        help='If the data is stored in subfolders')
    parser.add_argument('--output_dir', type=str, 
                        default='output', 
                        help='Dataset to use')
    parser.add_argument('--repo_id', type=str,
                        default="SmilingWolf/wd-eva02-large-tagger-v3",
                        help='Repository ID')
    parser.add_argument('--instance_token', type=str,
                        default="imgsks",
                        help='Instance token')
    parser.add_argument('--class_name', type=str,
                        default="woman",
                        help='Class name to use')
    parser.add_argument('--regular_dir', type=str,
                        default=None,
                        help='Directory to store the regular data')
    parser.add_argument('--gen_regular', action='store_true',
                        default=False,
                        help='Generate regular data')
    parser.add_argument("--regnum_per_image", type=int,
                        default=1,
                        help="regularzition number per image")
    parser.add_argument('--init_new', action='store_true',
                        default=False,
                        help='Initialize new data')
    return parser.parse_args()

def prepare_data(source_dir, output_dir, regular_files, args):
    sname = source_dir.name
    instance = sname.split('-')[-1]
    image_path = output_dir / sname / 'images'

    count = 0
    for file in tqdm(source_dir.iterdir()):
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            shutil.copy(file, image_path / file.name)
            count += 1
    if count == 0:
        raise ValueError('No image files found in the source directory')
    
    if regular_files:
        reg_path = output_dir / sname / 'reg'
        if reg_path.exists():
            shutil.rmtree(reg_path, ignore_errors=True)
        reg_path.mkdir(parents=True, exist_ok=True)
        ridx = random.sample(range(len(regular_files)), count)
        for idx in ridx:
            shutil.copy(regular_files[idx], reg_path / regular_files[idx].name)
    
    undesired_tags = "1girl,1boy,1women,1man,1person,child,solo,asian"
    cmd = f'python sd-scripts/finetune/tag_images_by_wd14_tagger.py'
    cmd += f' --onnx --repo_id {args.repo_id} --batch_size 4'
    cmd += f' --remove_underscore --use_rating_tags_as_last_tag'
    cmd += f' --undesired_tags="{undesired_tags}"'
    cmd += f' {str(image_path)}'
    ret = os.system(cmd)
    return ret

black_list = [
    "eye", "lip", "nose", "ear", "mouth", "teeth", "tongue", "neck",
    "smile", 
]
def clean_caption(source_dir, output_dir, instance_token, class_name):
    pstr = r"|".join(black_list)
    pattern = re.compile(pstr, re.IGNORECASE)
    
    sname = source_dir.name
    image_path = output_dir / sname / 'images'
    concept = f"{instance_token} {class_name}"
    
    count = 0
    sdict = {}
    orign_prompts = []
    for file in image_path.iterdir():
        if not (file.is_file() and file.suffix.lower() in ['.txt']):
            continue
        with open(file, 'r') as f:
            lines = f.readlines()
        tokens = lines[0].strip().split(", ")
        orign_prompts.append((tokens.copy(), file.name))
        out_tokens = [f"photo of a {concept}"]
        for token in tokens:
            pres = pattern.search(token)
            if pres is not None:
                continue
            out_tokens.append(token)
            if token not in sdict:
                sdict[token] = 0
            sdict[token] += 1
        with open(file, 'w') as f:
            f.write(", ".join(out_tokens))
        count += 1
        
    slist = sdict.items()
    slist = sorted(slist, key=lambda x: x[1], reverse=True)
    with open(output_dir/ sname / f"tag_statis.log", 'w') as f:
        for item in slist:
            f.write(f"{item[0]}: {item[1]}\n")
    return orign_prompts, sdict

def generate_random_letters(k=5):
    letters = random.choices(string.ascii_lowercase, k=k)
    return ''.join(letters)

def generate_regular_prompt(source_dir, output_dir, orign_prompts, tag_statis, args):
    pattern1 = re.compile(r"hair|bald", re.IGNORECASE)
    sname = source_dir.name
    instance = sname.split('-')[-1]
    image_path = output_dir / sname / 'images'
    reg_path = output_dir / sname / 'reg'
    reg_prompt_file = output_dir / sname / 'reg_prompt.txt'
    concept = f"{instance} {args.class_name}"
    
    image_count = len(orign_prompts)
    tag_list = tag_statis.items()
    tag_list = sorted(tag_list, key=lambda x: x[1], reverse=True)
    blist = []
    for item in tag_list:
        if item[1] > 0.8 * image_count:
            blist.append(item[0])
    
    reg_prompts = []
    for tokens, fname in orign_prompts:
        rand_name = generate_random_letters()
        reg_prompt = f"photo of a {rand_name} {args.class_name}"
        rtokens = []
        for token in tokens:
            if token in blist:
                continue
            if pattern1.search(token) is not None:
                continue
            rtokens.append(token)
        reg_prompt += ", "
        reg_prompt += ", ".join(rtokens)
        reg_prompts.append(reg_prompt)
    with open(reg_prompt_file, 'w') as f:
        f.write("\n".join(reg_prompts))

def generate_regular_images(source_dir, output_dir, args):
    sname = source_dir.name
    instance = sname.split('-')[-1]
    reg_path = output_dir / sname / 'reg'
    reg_prompt_file = output_dir / sname / 'reg_prompt.txt'
    with open(reg_prompt_file, 'r') as f:
        reg_prompts = f.readlines()
    for i in range(len(reg_prompts)):
        caption_file = reg_path / f'im_{i+1:06d}.txt'
        with open(caption_file, 'w') as f:
            si = reg_prompts[i].find(args.class_name)
            caption = "a "
            caption += reg_prompts[i].strip()[si:]
            f.write(caption+"\n")
        
    cmd = "python sd-scripts/sdxl_gen_img.py"
    cmd += " --ckpt /home/wangyh/sdxl_models/checkpoint/sd_xl_base_1.0.safetensors"
    cmd += " --vae /home/wangyh/sdxl_models/vae/madebyollin-sdxl-vae-fp16-fix.safetensors"
    cmd += f" --outdir {str(reg_path)}"
    cmd += f" --xformers --fp16"
    cmd += f" --W 1024 --H 1024"
    cmd += f" --steps 30 --scale 10.0 --seed=4334"
    cmd += f" --images_per_prompt {args.regnum_per_image}"
    cmd += f" --from_file \"{str(reg_prompt_file)}\""
    cmd += f" --sequential_file_name"
    cmd += f" --sampler \"dpmsolver++\""
    ret = os.system(cmd)
    return ret
            
def create_dataset_toml(source_dir, output_dir, args):
    sname = source_dir.name
    instance = sname.split('-')[-1]
    image_path = output_dir / sname / 'images'
    reg_path = output_dir / sname / 'reg'
    config_path = output_dir / sname / 'config'
    config_file = config_path / 'data_set.toml'
    
    config_dict = {}
    config_dict['general'] = {
        # 'shuffle_caption': True,
        'caption_extension': '.txt',
        'keep_tokens': 2,
    }
    config_dict['datasets'] = [
        {
        'resolution': 1024,
        # 'batch_size': 4,
        'keep_tokens': 2,
        'subsets':[
            {
            'image_dir': f'{str(image_path.resolve())}',
            'class_tokens': f'{instance} {args.class_name}',
            'num_repeats': 4,
            'keep_tokens': 2,
            }
        ]
        }
    ]
    if args.regular_dir or args.gen_regular:
        config_dict['datasets'][0]['subsets'].append({
            'is_reg': True,
            'image_dir': f'{str(reg_path.resolve())}',
            'class_tokens': f'{args.class_name}',
            'num_repeats': 1,
            'keep_tokens': 1,
        })
    toml.dump(config_dict, open(config_file, 'w'))

def init_folder(source_dir, output_dir, args):
    sname = source_dir.name
    instance = sname.split('-')[-1]
    image_path = output_dir / sname / 'images'
    if args.init_new and image_path.exists():
        shutil.rmtree(image_path, ignore_errors=True)
    image_path.mkdir(parents=True, exist_ok=True)
    reg_path = output_dir / sname / 'reg'
    if args.init_new and reg_path.exists():
        shutil.rmtree(reg_path, ignore_errors=True)
    reg_path.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / sname / 'config'
    if args.init_new and config_path.exists():
        shutil.rmtree(config_path, ignore_errors=True)
    config_path.mkdir(parents=True, exist_ok=True)
    
def main(args):
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    regular_dir = None if args.regular_dir is None else Path(args.regular_dir)
    
    dir_list = []
    if args.subfolders:
        for folder in source_dir.iterdir():
            if folder.is_dir():
                dir_list.append(folder)
    else:
        dir_list.append(source_dir)
    dir_list = sorted(dir_list, key=lambda x: x.name)
    
    regular_files = []
    if regular_dir:
        for file in regular_dir.iterdir():
            if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                regular_files.append(file)
    regular_files = sorted(regular_files, key=lambda x: x.name)
    
    for folder in dir_list:
        init_folder(folder, output_dir, args)
        ret = prepare_data(folder, output_dir, regular_files, args)
        if ret != 0:
            raise ValueError(f"Error in {folder.name}")
        orign_prompts, tag_statis = clean_caption(
            folder, output_dir, args.instance_token, args.class_name)
        if args.gen_regular:
            generate_regular_prompt(folder, output_dir, orign_prompts, tag_statis, args)
            ret = generate_regular_images(folder, output_dir, args)
            if ret != 0:
                raise ValueError(f"Error in {folder.name}")
        create_dataset_toml(folder, output_dir, args)

if __name__ == '__main__':
    args = argument_parse()
    main(args) 