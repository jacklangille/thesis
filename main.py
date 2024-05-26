import torch
from torchvision.utils import save_image
from modules.utils.base_eval import evaluate
from modules.utils.load_models import (
    load_base_resnet,
    load_base_squeezenet,
    load_fx_resnet,
    load_fx_squeezenet,
)
from modules.utils.data_gen_v2 import make_regular_loaders, make_noise_loader

device = "cuda" if torch.cuda.is_available() else "cpu"


def model_setup(bitsW=8, bitsA=8):
    resnet_base = load_base_resnet()
    squeezenet_base = load_base_squeezenet()
    resnet_quant = load_fx_resnet(bitsW, bitsA)
    squeezenet_quant = load_fx_squeezenet(bitsW, bitsA)
    return resnet_base, squeezenet_base, resnet_quant, squeezenet_quant


def im_save(dataloader):
    for images, labels in dataloader:
        save_image(images[0], "image.png")
        break


if __name__ == "__main__":
    _, _, test_loader = make_regular_loaders()

    noise_loader = make_noise_loader(
        64, "colored", snr_db=5, beta=0, noise_density=None, batch_size=32
    )
    im_save(noise_loader)
    resnet_base, _, resnet_quant, _ = model_setup(8, 8)
