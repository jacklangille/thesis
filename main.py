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


def compute_jacobian(model, loader, num_batches = 1, device):
    model.to(device)
    model.eval()
    softmax = torch.nn.Softmax(dim=1)  
    batch_counter = 0
    for inputs, _ in test_loader:
        if batch_counter >= num_batches_to_process:
            break
        
        inputs = inputs.to(device)
        inputs.requires_grad_(True)
        outputs = resnet_base(inputs)
        probabilities = softmax(outputs)  
        batch_size = inputs.size(0)
        num_classes = probabilities.size(1)
        jacobian_batch = torch.zeros(batch_size, num_classes, *inputs.size()[1:])
        frobenius_norms = torch.zeros(batch_size)
        for i in range(num_classes):
            resnet_base.zero_grad()
            if i > 0:
                inputs.grad.zero_()
            probabilities[:, i].sum().backward(retain_graph=True)
            jacobian_batch[:, i] = inputs.grad.data
        
        for j in range(batch_size):
            frobenius_norms[j] = torch.norm(jacobian_batch[j], p='fro')
        batch_counter += 1 

if __name__ == "__main__":
    _, _, test_loader = make_regular_loaders()
    resnet_base, _, resnet_quant, _ = model_setup(8,8) 

    
