# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from typing import Dict, Tuple

from torch import nn

from .detr import build


def build_model(args: argparse.Namespace) -> Tuple[nn.Module, nn.Module, Dict[str, nn.Module]]:
    """Returns: (model, criterion, postprocessors)"""
    return build(args)
