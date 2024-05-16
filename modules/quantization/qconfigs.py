import torch
import torch.quantization as tq
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization._learnable_fake_quantize import (
    _LearnableFakeQuantize as LearnableFakeQuantize,
)


def make_qconfigs(bitsW, bitsA):
    quant_min_act = 0
    quant_max_act = 2 ** (bitsA) - 1
    quant_min_weight = -(2 ** (bitsW - 1))
    quant_max_weight = 2 ** (bitsW - 1) - 1

    learnable_act = lambda range: LearnableFakeQuantize.with_args(
        observer=tq.HistogramObserver,
        quant_min=quant_min_act,
        quant_max=quant_max_act,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        scale=range / quant_max_act,
        zero_point=0.0,
        use_grad_scaling=True,
    )

    learnable_weights = lambda channels: LearnableFakeQuantize.with_args(
        observer=tq.PerChannelMinMaxObserver,
        quant_min=quant_min_weight,
        quant_max=quant_max_weight,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        scale=0.1,
        zero_point=0.0,
        use_grad_scaling=True,
        channel_len=channels,
    )

    fake_quant_act = FakeQuantize.with_args(
        observer=tq.HistogramObserver.with_args(
            quant_min=quant_min_act,
            quant_max=quant_max_act,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
        ),
    )

    return learnable_act, learnable_weights, fake_quant_act
