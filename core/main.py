import torch

from modules.utils.base_eval import evaluate
from modules.utils.load_models import (
    load_base_resnet,
    load_base_squeezenet,
    load_fx_resnet,
    load_fx_squeezenet,
)
from modules.utils.data_gen import make_regular_loaders

device = "cuda" if torch.cuda.is_available() else "cpu"

def model_setup(bitsW=8, bitsA=8):
    resnet_base = load_base_resnet()
    squeezenet_base = load_base_squeezenet()
    resnet_quant = load_fx_resnet(bitsW, bitsA)
    squeezenet_quant = load_fx_squeezenet(bitsW, bitsA)
    return resnet_base, squeezenet_base, resnet_quant, squeezenet_quant

def experiment(model):
    _, _, test_loader = make_regular_loaders()
    return evaluate(model, test_loader, "cpu")

if __name__ == "__main__":
    resnet_base, _, _, _ = model_setup(8, 8)
    results = experiment(resnet_base)


