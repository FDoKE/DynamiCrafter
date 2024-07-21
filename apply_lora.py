import argparse
import os
from collections import OrderedDict

from safetensors.torch import load_file
from safetensors.torch import save_file
import torch


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="model.ckpt",
                        help="select base model to which lora will be applied")
    parser.add_argument("--lora", type=str, default="lora.ckpt", help="select lora to apply")
    parser.add_argument("--alpha", type=float, default="1.0", help="alpha value of applied lora (0 = will leave base model weights, 1 = will apply lora fully)")
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

    path_to_base_model = args.base_model
    model_state_dict = load_state_dict(path_to_base_model)

    path_to_lora = args.lora
    lora_state_dict = load_state_dict(path_to_lora)

    state_dict_result = OrderedDict()
    for lora_parameter in lora_state_dict.keys():
        if "lora_a" in lora_parameter:
            lora_a = lora_state_dict[lora_parameter]
            lora_b = lora_state_dict[lora_parameter.replace('lora_a', 'lora_b')]
            model_weight_name = lora_parameter.replace('lora_a.', '')
            model_weight = model_state_dict[model_weight_name]
            new_weight = model_weight + (lora_b @ lora_a) * args.alpha
            model_state_dict[model_weight_name] = new_weight

    basename = os.path.basename(path_to_base_model)
    basename = basename[:basename.rindex(".")]
    result_name = basename + '_merged.' + args.format

    if args.format == 'safetensors':
        save_file(model_state_dict, result_name)
    else:
        torch.save(model_state_dict, result_name)

    print("Saved ", result_name)
