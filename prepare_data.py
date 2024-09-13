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
        t_data.update(cfg_data["prepare"])
        cfg_args = OmegaConf.create(t_data)
        return cfg_args
    

def prepare_data(source_dir, work_dir, cfg_args):
    logging.info(f"caption image for {source_dir.name}")
    instance_token = cfg_args.instance_token
    image_path = work_dir / source_dir.name / cfg_args.images_dir_name

    count = 0
    for file in tqdm(source_dir.iterdir()):
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            shutil.copy(file, image_path / file.name)
            count += 1
    if count == 0:
        raise ValueError('No image files found in the source directory')
    
    undesired_tags = "1girl,1boy,1women,1man,1person,child,solo,asian,chinese,japanese,korean"
    cmd = f'python sd-scripts/finetune/tag_images_by_wd14_tagger.py'
    cmd += f' --onnx --repo_id {cfg_args.repo_id} --batch_size 4'
    cmd += f' --remove_underscore --use_rating_tags_as_last_tag'
    cmd += f' --undesired_tags="{undesired_tags}"'
    cmd += f' {str(image_path)}'
    ret = os.system(cmd)
    return ret

def rewrite_caption(source_dir, work_dir, cfg_args):
    black_list = [
        "eye", "lip", "nose", "ear", "mouth", "teeth", "tongue", "neck", "hair",
    ]
    image_path = work_dir / source_dir.name / cfg_args.images_dir_name
    pstr = r"|".join(black_list)
    pattern = re.compile(pstr, re.IGNORECASE)
    concept = f"{cfg_args.instance_token} {cfg_args.class_name}"
    if cfg_args.additional_caption is not None:
        concept += f", {cfg_args.additional_caption}"
    
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
    with open(image_path.parent / f"tag_statis.log", 'w') as f:
        for item in slist:
            f.write(f"{item[0]}: {item[1]}\n")
    return orign_prompts, sdict


def prepare_class_data(source_dir, work_dir, cfg_args):
    logging.info(f"prepare class data for {source_dir.name}")
    source_class_data_dir = Path(cfg_args.source_class_data_dir)
    class_data_num = cfg_args.class_data_num
    class_data_dir = work_dir / source_dir.name / cfg_args.class_data_dir_name
    if cfg_args.init_new and class_data_dir.exists():
        shutil.rmtree(class_data_dir, ignore_errors=True)
    class_data_dir.mkdir(parents=True, exist_ok=True)
    
    file_list = []
    for file in tqdm(source_class_data_dir.iterdir()):
        if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            file_list.append(file)
    random.shuffle(file_list)
    for file in file_list[:class_data_num]:
        shutil.copy(file, class_data_dir / file.name)


def create_dataset_toml(source_dir, work_dir, cfg_args):
    sname = source_dir.name
    instance_token = cfg_args.instance_token
    image_path = work_dir / sname / cfg_args.images_dir_name
    config_path = work_dir / sname / 'config'
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
            'class_tokens': f'{instance_token} {cfg_args.class_name}',
            'num_repeats': 1,
            'keep_tokens': 2,
            }
        ]
        }
    ]
    if cfg_args.with_class_data:
        class_data_dir = work_dir / source_dir.name / cfg_args.class_data_dir_name
        tdict = {
            "is_reg": True,
            "image_dir": str(class_data_dir.resolve()),
            "class_tokens": cfg_args.class_name,
            "keep_tokens": 1,
        }
        config_dict['datasets'][0]['subsets'].append(tdict)
    toml.dump(config_dict, open(config_file, 'w'))

def init_folder(source_dir, work_dir, cfg_args):
    sname = source_dir.name
    image_path = work_dir / sname / cfg_args.images_dir_name
    if cfg_args.init_new and image_path.exists():
        shutil.rmtree(image_path, ignore_errors=True)
    image_path.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / sname / 'config'
    if cfg_args.init_new and config_path.exists():
        shutil.rmtree(config_path, ignore_errors=True)
    config_path.mkdir(parents=True, exist_ok=True)


def main(cfg_args):
    if cfg_args.work_dir is None:
        raise ValueError("work_dir is not set")
    work_dir = Path(cfg_args.work_dir)
    source_dir = Path(cfg_args.instance_image_dir)
    
    source_dir_list = []
    if cfg_args.subfolders:
        for folder in source_dir.iterdir():
            if folder.is_dir():
                source_dir_list.append(folder)
    else:
        source_dir_list.append(source_dir)
    source_dir_list = sorted(source_dir_list, key=lambda x: x.name)
    
    for folder in source_dir_list:
        init_folder(folder, work_dir, cfg_args)
        ret = prepare_data(folder, work_dir, cfg_args)
        if ret != 0:
            raise ValueError(f"Error in {folder.name}")
        orign_prompts, tag_statis = rewrite_caption(
            folder, work_dir, cfg_args)
        if cfg_args.with_class_data:
            prepare_class_data(folder, work_dir, cfg_args)
        create_dataset_toml(folder, work_dir, cfg_args)

if __name__ == '__main__':
    args = parse_args()
    cfg_args = load_config_yaml(args)
    main(cfg_args)