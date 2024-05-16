import torch


def compute_jacobian(model, device, x):
    model.eval()
    model.to(device)
    x = x.to(device)

    x.requires_grad_(True)
    output = torch.softmax(model(x), dim=1)

    num_inputs = x.nelement()
    num_classes = output.size(1)

    jacobian = torch.zeros((num_classes, num_inputs))

    print(f"Output shape: {output.shape}")
    print(f"Jacobian shape: {jacobian.shape}")

    for i in range(num_classes):
        model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        output[0][i].backward(retain_graph=True)
        jacobian[i] = x.grad.flatten()
    return jacobian
