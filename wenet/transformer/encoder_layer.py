#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import torch
from torch import nn
from wenet.transformer.layer_norm import LayerNorm
from wenet.utils.common import set_seed

class TransformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)

    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: torch.nn.Module,
        dropout_rate: float,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        # concat_linear may be not used in forward fuction,
        # but will be saved in the *.pt
        if concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        output_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            pos_emb (torch.Tensor): just for interface compatibility
                to ConformerEncoderLayer
            mask_pad (torch.Tensor): does not used in transformer layer,
                just for unified api with conformer.
            output_cache (torch.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
            cnn_cache (torch.Tensor): not used here, it's for interface
                compatibility to ConformerEncoderLayer
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.size(0) == x.size(0)
            assert output_cache.size(2) == self.size
            assert output_cache.size(1) < x.size(1)
            chunk = x.size(1) - output_cache.size(1)
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        fake_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        return x, mask, fake_cnn_cache


class ConformerEncoderLayer1(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """
    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: Optional[torch.Tensor] = None,
        output_cache: Optional[torch.Tensor] = None,
        cnn_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time).
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time)
            output_cache (torch.Tensor): Cache tensor of the output
                (#batch, time2, size), time2 < time in x.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.size(0) == x.size(0)
            assert output_cache.size(2) == self.size
            assert output_cache.size(1) < x.size(1)
            chunk = x.size(1) - output_cache.size(1)
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        x_att = self.self_attn(x_q, x, x, mask, pos_emb)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.tensor([0.0], dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)

        return x, mask, new_cnn_cache




class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        set_seed(1)
        super(ConformerEncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = LayerNorm(size)  # for the FNN module
        self.norm_mha = LayerNorm(size)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module
            self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x, pos_emb, mask, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
            self.feed_forward(x)
        )
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x, pos_emb, mask
