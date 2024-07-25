import argparse
import os
import tkinter as tk
from tkinter import filedialog, simpledialog

import torch
from safetensors.torch import load_file
from safetensors.torch import save_file

root = tk.Tk()
root.withdraw()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=str, default="dialog",
                        help="first model to merge")
    parser.add_argument("--right", type=str, default="dialog", help="second model to merge")
    parser.add_argument("--alpha", type=str, default="dialog",
                        help="0.0 - left weights will be used, 1.0 - right weights will be used, other will merge accordingly (1.0-A)*left+A*right")
    parser.add_argument("--format", type=str, default="safetensors", choices=["safetensors", "ckpt"],
                        help="result format (can be ckpt or safetensors)")

    return parser


def load_state_dict(path_to_model):
    if ".safetensors" in path_to_model:
        try:
            state_dict = load_file(path_to_model)
        except:
            print("failed to load in safetensors format, load as pytorch")
            state_dict = torch.load(path_to_model, map_location="cpu")
    else:
        state_dict = torch.load(path_to_model, map_location="cpu")

    while "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]

    return state_dict


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    try:
        path_to_left = args.left
        left_state_dict = load_state_dict(path_to_left)
    except:
        path_to_left = filedialog.askopenfilename()
        left_state_dict = load_state_dict(path_to_left)

    try:
        path_to_right = args.right
        right_state_dict = load_state_dict(path_to_right)
    except:
        path_to_right = filedialog.askopenfilename()
        right_state_dict = load_state_dict(path_to_right)

    try:
        alpha = float(args.alpha)
    except:
        alpha = simpledialog.askfloat(title="Pick", initialvalue=0.5, prompt="Alpha value")

    right_keys = right_state_dict.keys()
    left_keys = left_state_dict.keys()
    for layer in left_keys:
        if layer in right_keys:
            left_state_dict[layer] = (1.0 - alpha) * left_state_dict[layer] + alpha * right_state_dict[layer]
    for layer in right_keys:
        if layer not in left_keys:
            left_state_dict[layer] = right_state_dict[layer]
            print(f"not present layer {layer} adding as is")

    leftName = os.path.basename(path_to_left)
    leftName = leftName[:leftName.rindex(".")]
    rightName = os.path.basename(path_to_right)
    rightName = rightName[:rightName.rindex(".")]
    result_name = leftName + "_" + rightName + '_merged.' + args.format

    if args.format == 'safetensors':
        save_file(left_state_dict, result_name)
    else:
        torch.save(left_state_dict, result_name)

    print("Saved ", result_name)
