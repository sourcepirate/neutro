"""Layer implementations, grouped by family.

The beginner path is ``Dense`` -> ``Dropout`` -> ``Flatten`` -> ``Sequential``.
Advanced families (attention, transformer, diffusion, fine-tuning) live in
their own subpackages and are re-exported here for convenience.
"""

from .base import Layer
from .core.input_layer import Input, InputLayer
from .core.dense import Dense
from .core.dropout import Dropout
from .core.flatten import Flatten
from .core.activation import Activation, ReLU, Softmax, Sigmoid, Tanh
from .core.bitlinear import BitLinear
from .core.reparameterization import Reparameterization
from .core.moe import MoELayer
from .core.swiglu_mlp import SwiGLUMLP
from .core.lora import LoRADense, apply_lora
from .core.qlora import QLoRADense, apply_qlora
from .core.merging import Add, Concatenate, Multiply, Average, Maximum, Minimum
from .convolutional.conv2d import Conv2D
from .convolutional.conv1d import Conv1D
from .pooling.maxpooling2d import MaxPooling2D
from .pooling.global_pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from .pooling.upsampling2d import UpSampling2D
from .recurrent.simple_rnn import SimpleRNN
from .recurrent.lstm import LSTM
from .recurrent.gru import GRU
from .attention.mha import MultiHeadAttention
from .attention.mqa import MultiQueryAttention
from .attention.gqa import GroupedQueryAttention
from .attention.flash_attention import FlashAttention
from .attention.paged_attention import PagedAttention, PagedKVCache
from .normalization.layernorm import LayerNormalization
from .normalization.batchnorm import BatchNormalization
from .normalization.rmsnorm import RMSNorm
from .normalization.groupnorm import GroupNormalization
from .normalization.qknorm import QKNorm
from .diffusion.adaln import AdaLN, AdaLNZero
from .diffusion.time_mlp import TimeMLPEmbedding
from .diffusion.dit_block import DiTBlock
from .diffusion.discrete_scheduler import DiscreteDiffusionScheduler
from .embedding.embedding import Embedding
from .embedding.time_embedding import TimeEmbedding
from .embedding.token_position_embedding import TokenPositionEmbedding
from .transformer.transformer_block import TransformerBlock
from .transformer.bitnet_block import BitNetBlock
from .transformer.qwen_block import QwenBlock
from .transformer.moe_block import MoETransformerBlock

__all__ = [
    "Layer",
    "Input", "InputLayer",
    "Dense", "Dropout", "Flatten",
    "Activation", "ReLU", "Softmax", "Sigmoid", "Tanh",
    "BitLinear", "Reparameterization",
    "MoELayer", "SwiGLUMLP",
    "LoRADense", "apply_lora", "QLoRADense", "apply_qlora",
    "Add", "Concatenate", "Multiply", "Average", "Maximum", "Minimum",
    "Conv2D", "Conv1D",
    "MaxPooling2D", "GlobalAveragePooling2D", "GlobalMaxPooling2D",
    "UpSampling2D",
    "SimpleRNN", "LSTM", "GRU",
    "MultiHeadAttention", "MultiQueryAttention", "GroupedQueryAttention",
    "FlashAttention", "PagedAttention", "PagedKVCache",
    "LayerNormalization", "BatchNormalization", "RMSNorm",
    "GroupNormalization", "QKNorm",
    "AdaLN", "AdaLNZero", "TimeMLPEmbedding", "DiTBlock",
    "DiscreteDiffusionScheduler",
    "Embedding", "TimeEmbedding", "TokenPositionEmbedding",
    "TransformerBlock", "BitNetBlock", "QwenBlock", "MoETransformerBlock",
]
