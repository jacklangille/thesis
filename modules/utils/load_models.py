import copy
import torch
from modules.model_defs.resnet import resnet18
from modules.model_defs.squeeze import squeezenet1_1
from modules.quantization.qconfigs import make_qconfigs
import torch
import torch.quantization as tq
from torch.ao.quantization._learnable_fake_quantize import (
    _LearnableFakeQuantize as LearnableFakeQuantize,
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.ao.quantization.qconfig_mapping import QConfigMapping


def load_base_squeezenet(num_classes=70):
    model_path = "squeezenet/squeezenet_epoch_70.pth"
    model = squeezenet1_1()
    model.classifier[1] = torch.nn.Conv2d(
        512, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )
    model.classifier[3] = torch.nn.AdaptiveAvgPool2d((1, 1))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_base_resnet(num_classes=70):
    model_path = "resnet/resnet_epoch_70.pth"
    model = resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_fx_squeezenet(bitsW, bitsA):
    raw_model = load_base_squeezenet()
    quant_model = copy.deepcopy(raw_model)
    learnable_act, learnable_weights, fake_quant_act = make_qconfigs(bitsW, bitsA)

    qconfig_global = tq.QConfig(
        activation=fake_quant_act, weight=tq.default_fused_per_channel_wt_fake_quant
    )

    qconfig_mapping = QConfigMapping().set_global(qconfig_global)

    for name, module in quant_model.named_modules():
        if hasattr(module, "out_channels"):
            qconfig = tq.QConfig(
                activation=learnable_act(range=2),
                weight=learnable_weights(channels=module.out_channels),
            )
            qconfig_mapping.set_module_name(name, qconfig)

    example_inputs = (torch.randn(1, 3, 224, 224),)
    quant_model.eval()
    fx_model = prepare_qat_fx(quant_model, qconfig_mapping, example_inputs)

    return fx_model


def load_fx_resnet(bitsW, bitsA):
    raw_model = load_base_resnet()
    quant_model = copy.deepcopy(raw_model)
    learnable_act, learnable_weights, fake_quant_act = make_qconfigs(bitsW, bitsA)

    qconfig_global = tq.QConfig(
        activation=fake_quant_act, weight=tq.default_fused_per_channel_wt_fake_quant
    )

    qconfig_mapping = QConfigMapping().set_global(qconfig_global)

    for name, module in quant_model.named_modules():
        if hasattr(module, "out_channels"):
            qconfig = tq.QConfig(
                activation=learnable_act(range=2),
                weight=learnable_weights(channels=module.out_channels),
            )
            qconfig_mapping.set_module_name(name, qconfig)

    example_inputs = (torch.randn(1, 3, 224, 224),)
    quant_model.eval()
    fx_model = prepare_qat_fx(quant_model, qconfig_mapping, example_inputs)

    return fx_model
