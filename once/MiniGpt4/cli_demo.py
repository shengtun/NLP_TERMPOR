# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple command-line interactive chat demo."""

import argparse
import os
import platform
import shutil
from copy import deepcopy

import torch
from lavis.models import load_model_and_preprocess
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed

from functools import partial
from PIL import Image

_WELCOME_MSG = '''\
Welcome to use MiniGPT4Qwen(based on LAVIS, MiniGPT4 and Qwen-Chat model), type text to start chat, type :h to show command help.
(Welcome to the Qwen-Chat model, you can enter the content to chat, :h display command help)

Note: This demo is governed by the original license of Qwen.
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc.
'''
_HELP_MSG = '''\
Commands:
    :help / :h          Show this help message             
    :exit / :quit / :q  Exit the demo                   
    :clear / :cl        Clear screen                        
    :clear-his / :clh   Clear history                     
    :history / :his     Show history                       

'''


def _load_model_processor(args):
    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "cuda"
    
    global load_model_and_preprocess
    load_model_and_preprocess = partial(load_model_and_preprocess,is_eval=True,device=device_map)

    model, vis_processors, _ = load_model_and_preprocess("minigpt4qwen", args.model_type,llm_device_map=args.llm_device_map)
    model.load_checkpoint(args.checkpoint_path)

    model.llm_model.transformer.bfloat16()
    model.llm_model.lm_head.bfloat16()

    generation_config = {
    "chat_format": "chatml",
    "eos_token_id": 151643,
    "pad_token_id": 151643,
    "max_window_size": 6144,
    "max_new_tokens": 512,
    "do_sample": False,
    "transformers_version": "4.31.0"
    }

    generation_config = GenerationConfig.from_dict(generation_config)

    return model, vis_processors, generation_config


def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clear_screen():
    if platform.system() == "Windows":
        os.system("cls")
    else:
        os.system("clear")


def _print_history(history):
    terminal_width = shutil.get_terminal_size()[0]
    print(f'History ({len(history)})'.center(terminal_width, '='))
    for index, (query, response) in enumerate(history):
        print(f'User[{index}]: {query}')
        print(f'QWen[{index}]: {response}')
    print('=' * terminal_width)

# voice massage input here
def _get_input() -> str:
    while True:
        try:
            message = input('User> ').strip()
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input')
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            return message
        print('[ERROR] Query is empty')

def _get_image_input():
    images, messages = [], []
    while True:
        try:
            message = input('Please input the path of images (you can input multiple image paths, use `:f` to finish input):> ')
        except UnicodeDecodeError:
            print('[ERROR] Encoding error in input')
            continue
        except KeyboardInterrupt:
            exit(1)
        if message:
            if message == ":f":
                print("[Finished] You've finished to input the images!")
                return images, messages
            try:
                image = Image.open(message).convert("RGB")
            except Exception as e:
                print(e)
                continue
            images.append(image)
            messages.append(message)
        else:
            print('[ERROR] Query is empty')



def main():
    parser = argparse.ArgumentParser(
        description='QWen-Chat command-line interactive chat demo.')
    parser.add_argument("--model-type",type=str,default='qwen7b_chat',choices=['qwen7b_chat','qwen14b_chat'])
    parser.add_argument("-c", "--checkpoint-path", type=str,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--llm_device_map", type=str, default="cpu")
    args = parser.parse_args()

    history, response = [], ''

    model, vis_processors, generation_config = _load_model_processor(args)
    orig_gen_config = deepcopy(model.llm_model.generation_config)

    images, image_paths = _get_image_input()
    image_tensor = torch.stack([vis_processors['eval'](image) for image in images], dim=0).to(model.device)

    _clear_screen()
    print(_WELCOME_MSG)

    seed = args.seed

    first = True
    while True:
        query = _get_input()
        
        if not history:
            first = True

        # Process commands.
        if query.startswith(':'):
            command_words = query[1:].strip().split()
            if not command_words:
                command = ''
            else:
                command = command_words[0]

            if command in ['exit', 'quit', 'q']:
                break
            elif command in ['clear', 'cl']:
                _clear_screen()
                print(_WELCOME_MSG)
                _gc()
                continue
            elif command in ['clear-history', 'clh']:
                print(f'[INFO] All {len(history)} history cleared')
                history.clear()
                _gc()
                continue
            elif command in ['help', 'h']:
                print(_HELP_MSG)
                continue
            elif command in ['history', 'his']:
                _print_history(history)
                continue
            elif command in ['seed']:
                if len(command_words) == 1:
                    print(f'[INFO] Current random seed: {seed}')
                    continue
                else:
                    new_seed_s = command_words[1]
                    try:
                        new_seed = int(new_seed_s)
                    except ValueError:
                        print(f'[WARNING] Fail to change random seed: {new_seed_s!r} is not a valid number')
                    else:
                        print(f'[INFO] Random seed changed to {new_seed}')
                        seed = new_seed
                    continue
            elif command in ['conf']:
                if len(command_words) == 1:
                    print(model.llm_model.generation_config)
                else:
                    for key_value_pairs_str in command_words[1:]:
                        eq_idx = key_value_pairs_str.find('=')
                        if eq_idx == -1:
                            print('[WARNING] format: <key>=<value>')
                            continue
                        conf_key, conf_value_str = key_value_pairs_str[:eq_idx], key_value_pairs_str[eq_idx + 1:]
                        try:
                            conf_value = eval(conf_value_str)
                        except Exception as e:
                            print(e)
                            continue
                        else:
                            print(f'[INFO] Change config: model.llm_model.generation_config.{conf_key} = {conf_value}')
                            setattr(model.llm_model.generation_config, conf_key, conf_value)
                continue
            elif command in ['reset-conf']:
                print('[INFO] Reset generation config')
                model.llm_model.generation_config = deepcopy(orig_gen_config)
                print(model.llm_model.generation_config)
                continue
            elif command in ['img']:
                print(f'[INFO] Image Path: {image_paths}')
                continue
            else:
                # As normal query.
                pass

        # Run chat. put whisper in here
        set_seed(seed)
        try:
            if first:
                if '<ImageHere>' not in query:
                    # query = f'<Img>{"<ImageHere>" * len(image_paths)}</Img> ' + query
                    img_query = ""
                    for _ in image_paths:
                        img_query += '<Img><ImageHere></Img>'
                    query = img_query + query
                first = False
            if args.cpu_only:
                model.bfloat16()
                response, history = model.chat(query, history=history, image_tensor=image_tensor.bfloat16(), generation_config=generation_config)
            else:
                with torch.cuda.amp.autocast(enabled=True,dtype=torch.bfloat16):
                    response, history = model.chat(query, history=history, image_tensor=image_tensor, generation_config=generation_config)
            _clear_screen()
            print(f"\n\033[33mUser:\033[0m {query}")
            print(f"\n\033[31mMPP-Qwen:\033[0m {response}")
        except KeyboardInterrupt:
            print('[WARNING] Generation interrupted')
            continue



if __name__ == "__main__":
    main()