import argparse
import os
import tkinter as tk
from collections import OrderedDict
from tkinter import filedialog, simpledialog

import torch
from safetensors.torch import save_file

from utils.lora_utils import load_state_dict

root = tk.Tk()
root.withdraw()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="dialog",
                        help="select base model to which lora will be applied")
    parser.add_argument("--lora", type=str, default="dialog", help="select lora to apply")
    parser.add_argument("--alpha", type=str, default="dialog",
                        help="alpha value of applied lora (0 = will leave base model weights, 1 = will apply lora fully)")
    parser.add_argument("--format", type=str, default="safetensors", choices=["safetensors", "ckpt"],
                        help="result format (can be ckpt or safetensors)")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    try:
        path_to_base_model = args.base_model
        model_state_dict = load_state_dict(path_to_base_model)
    except:
        path_to_base_model = filedialog.askopenfilename(title="SELECT BASE MODEL")
        model_state_dict = load_state_dict(path_to_base_model)

    try:
        path_to_lora = args.lora
        lora_state_dict = load_state_dict(path_to_lora)
    except:
        path_to_lora = filedialog.askopenfilename(title="SELECT LORA MODEL OR CHECKPOINT CONTAINING LORA")
        lora_state_dict = load_state_dict(path_to_lora)

    try:
        alpha = float(args.alpha)
    except:
        alpha = simpledialog.askfloat(title="Pick", initialvalue=1.0, prompt="Alpha value")

    state_dict_result = OrderedDict()
    for lora_parameter in lora_state_dict.keys():
        if "lora_a" in lora_parameter:
            lora_a = lora_state_dict[lora_parameter]
            lora_b = lora_state_dict[lora_parameter.replace('lora_a', 'lora_b')]
            model_weight_name = lora_parameter.replace('lora_a.', '')
            model_weight = model_state_dict[model_weight_name]
            new_weight = model_weight + (lora_b @ lora_a) * alpha
            model_state_dict[model_weight_name] = new_weight

    modelbasename = os.path.basename(path_to_base_model)
    modelbasename = modelbasename[:modelbasename.rindex(".")]
    basename = os.path.basename(path_to_lora)
    basename = basename[:basename.rindex(".")]
    result_name = modelbasename + '_' + basename + '_applied.' + args.format

    if args.format == 'safetensors':
        save_file(model_state_dict, result_name)
    else:
        torch.save(model_state_dict, result_name)

    print("Saved ", result_name)
