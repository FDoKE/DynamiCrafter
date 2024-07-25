from safetensors.torch import load_file
import torch

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