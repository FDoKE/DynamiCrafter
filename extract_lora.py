import argparse
import os
from collections import OrderedDict

from safetensors.torch import load_file
from safetensors.torch import save_file
import torch
import tkinter as tk
from tkinter import filedialog

from utils.lora_utils import load_state_dict

root = tk.Tk()
root.withdraw()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="dialog", help="select model to extract lora weights")
    parser.add_argument("--format", type=str, default="safetensors", choices=["safetensors", "ckpt"],
                        help="result format (can be ckpt or safetensors)")

    return parser

def extractLora(state_dict, path_to_model):
    state_dict_lora = OrderedDict()
    for layer_name in state_dict.keys():
        if "lora_" in layer_name:
            # print(layer_name)
            state_dict_lora[layer_name] = state_dict[layer_name]
    basename = os.path.basename(path_to_model)
    basename = basename[:basename.rindex(".")]
    result_name = basename + '_lora.' + args.format
    if args.format == 'safetensors':
        save_file(state_dict_lora, result_name)
    else:
        torch.save(state_dict_lora, result_name)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    path_to_model = args.model

    state_dict = None
    try:
        path_to_model = args.model
        state_dict = load_state_dict(path_to_model)
    except:
        path_to_models = filedialog.askopenfilenames(title="Pick models with LoRA to extract")

    if state_dict is None:
        for path_to_model in path_to_models:
            state_dict = load_state_dict(path_to_model)
            extractLora(state_dict, path_to_model)
    else:
        extractLora(state_dict, path_to_model)

    print("Saved ")
