#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""Subsampling layer definition."""

from typing import Tuple

import torch
from wenet.utils.common import set_seed

class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-12),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 5, 3),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(odim * (((idim - 1) // 2 - 2) // 3),
                                      odim)
        self.pos_enc = pos_enc_class
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.linear = torch.nn.Linear(
            odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def forward(
            self,
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]

class VGG2L(BaseSubsampling):
    """VGG-like module
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        super().__init__()
        set_seed(1)
        self.vgg2l = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((3, 2)),
            torch.nn.Conv2d(64, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((2, 2)),
        )
        #self.output = torch.nn.Linear(128 * ((idim // 2) // 2), odim)
        #self.pos_enc = pos_enc_class
        if pos_enc_class is not None:
            self.output = torch.nn.Sequential(
                torch.nn.Linear(128 * ((idim // 2) // 2), odim), pos_enc_class
            )
        else:
            self.output = torch.nn.Linear(128 * ((idim // 2) // 2), odim)

    def forward(self, 
            x: torch.Tensor,
            x_mask: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """VGG2L forward
        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """

        x = x.unsqueeze(1)  # (b, c, t, f)
        vgg_output = self.vgg2l(x)
        b, c, t, f = vgg_output.size()
        vgg_output = self.output(vgg_output.transpose(1, 2).contiguous().view(b, t, c * f))
        #x, pos_emb = self.pos_enc(x, offset)

        x_mask = self.create_new_mask(x_mask)

        return vgg_output[0], vgg_output[1], x_mask

    def create_new_mask(self, feats_mask: torch.Tensor) -> torch.Tensor:
        """Create a subsampled mask of feature sequences.

        Args:
            feats_mask: Mask of feature sequences. (B, 1, F)

        Returns:
            vgg_mask: Mask of VGG2L output sequences. (B, 1, sub(F))

        """
        vgg1_t_len = feats_mask.size(2) - (feats_mask.size(2) % 3)
        vgg_mask = feats_mask[:, :, :vgg1_t_len][:, :, ::3]

        vgg2_t_len = vgg_mask.size(2) - (vgg_mask.size(2) % 2)
        vgg_mask = vgg_mask[:, :, :vgg2_t_len][:, :, ::2]

        return vgg_mask
