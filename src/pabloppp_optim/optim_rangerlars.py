####
# CODE TAKEN FROM https://github.com/mgrankin/over9000
####

import itertools as it
import math

import torch
from lookahead import Lookahead
from ralamb import Ralamb
from torch.optim.optimizer import Optimizer

# RAdam + LARS + LookAHead

# Lookahead implementation from https://github.com/lonePatient/lookahead_pytorch/blob/master/optimizer.py
# RAdam + LARS implementation from https://gist.github.com/redknightlois/c4023d393eb8f92bb44b2ab582d7ec20


def Over9000(params, alpha=0.5, k=6, *args, **kwargs):
    ralamb = Ralamb(params, *args, **kwargs)
    return Lookahead(ralamb, alpha, k)


RangerLars = Over9000
