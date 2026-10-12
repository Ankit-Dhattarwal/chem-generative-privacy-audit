"""Opacus-compatible causal GPT used only for the DP-SGD arm."""
import math
import torch
from torch import nn
