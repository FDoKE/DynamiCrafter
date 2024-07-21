import argparse
import os
from collections import OrderedDict

from safetensors.torch import load_file
from safetensors.torch import save_file
import torch


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model.ckpt", help="select model to extract lora weights")
    parser.add_argument("--format", type=str, default="safetensors", choices=["safetensors", "ckpt"],
                        help="result format (can be ckpt or safetensors)")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    path_to_model = args.model
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

    state_dict_lora = OrderedDict()
    for layer_name in state_dict.keys():
        if "lora_" in layer_name:
            # print(layer_name)
            state_dict_lora[layer_name] = state_dict[layer_name]
    state_dict = None

    basename = os.path.basename(path_to_model)
    basename = basename[:basename.rindex(".")]
    result_name = basename + '_lora.' + args.format

    if args.format == 'safetensors':
        save_file(state_dict_lora, result_name)
    else:
        torch.save(state_dict_lora, result_name)

    print("Saved ", result_name)
