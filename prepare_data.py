import os, sys
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
import toml
import json
import re
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
    parser.add_argument('--class_name', type=str,
                        default="woman",
                        help='Class name to use')
    parser.add_argument('--regular_dir', type=str,
                        default=None,
                        help='Directory to store the regular data')
    parser.add_argument("--regular_name", type=str,
                        default=None,
                        help="regularzition prompt")
    parser.add_argument('--init_new', action='store_true',
                        default=False,
                        help='Initialize new data')
    return parser.parse_args()

def prepare_data(source_dir, output_dir, regular_files):
    sname = source_dir.name
    instance = sname.split('-')[-1]
    image_path = output_dir / sname / 'images'
    if args.init_new and image_path.exists():
        shutil.rmtree(image_path, ignore_errors=True)
    image_path.mkdir(parents=True, exist_ok=True)
    
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
        ridx = random.sample(range(len(regular_files)), 50)
        for idx in ridx:
            shutil.copy(regular_files[idx], reg_path / regular_files[idx].name)
    
    undesired_tags = "1girl,1boy,1women,1man,1person,solo,asian"
    cmd = f'python sd-scripts/finetune/tag_images_by_wd14_tagger.py'
    cmd += f' --onnx --repo_id SmilingWolf/wd-swinv2-tagger-v3 --batch_size 4'
    cmd += f' --remove_underscore --use_rating_tags_as_last_tag'
    cmd += f' --undesired_tags="{undesired_tags}"'
    cmd += f' {str(image_path)}'
    ret = os.system(cmd)
    return ret

black_list = [
    "eye", "lip", "nose", "ear", "mouth", "teeth", "tongue", "neck",
    "smile", 
]
def clean_caption(source_dir, output_dir, class_name):
    pstr = r"|".join(black_list)
    pattern = re.compile(pstr, re.IGNORECASE)
    
    sname = source_dir.name
    instance = sname.split('-')[-1]
    image_path = output_dir / sname / 'images'
    sdict = {}
    for file in image_path.iterdir():
        if not (file.is_file() and file.suffix.lower() in ['.txt']):
            continue
        with open(file, 'r') as f:
            lines = f.readlines()
        tokens = lines[0].strip().split(", ")
        out_tokens = [f"{instance} {class_name}"]
        for token in tokens:
            pres = pattern.search(token)
            if pres is not None:
                continue
            out_tokens.append(token)
            if token not in sdict:
                sdict[token] = 0
            sdict[token] += 1
            # btag = False
            # for bl in black_list:
            #     if token.lower().find(bl) != -1:
            #         btag = True
            #         break
            # if not btag:
            #     out_tokens.append(token)
        with open(file, 'w') as f:
            f.write(", ".join(out_tokens))
    slist = sdict.items()
    slist = sorted(slist, key=lambda x: x[1], reverse=True)
    with open(image_path / f"{sname}_tags.txt", 'w') as f:
        for item in slist:
            f.write(f"{item[0]}: {item[1]}\n")

def create_dataset_toml(source_dir, output_dir, args):
    sname = source_dir.name
    instance = sname.split('-')[-1]
    image_path = output_dir / sname / 'images'
    reg_path = output_dir / sname / 'reg'
    
    config_path = output_dir / sname / 'config'
    config_path.mkdir(parents=True, exist_ok=True)
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
            'num_repeats': 1,
            'keep_tokens': 2,
            }
        ]
        }
    ]
    if args.regular_dir:
        config_dict['datasets'][0]['subsets'].append({
            'is_reg': True,
            'image_dir': f'{str(reg_path.resolve())}',
            'class_tokens': f'{args.regular_name}',
            'num_repeats': 1,
            'keep_tokens': 1,
        })
    toml.dump(config_dict, open(config_file, 'w'))
    
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
        ret = prepare_data(folder, output_dir, regular_files)
        if ret != 0:
            raise ValueError(f"Error in {folder.name}")
        clean_caption(folder, output_dir, args.class_name)
        create_dataset_toml(folder, output_dir, args)

if __name__ == '__main__':
    args = argument_parse()
    main(args) 