
import torch
import torch.nn as nn
import torch.nn.functional as F

def feature_loss(real_features, fake_features):
    loss = 0
    for rf, ff in zip(real_features, fake_features):
        for r, f in zip(rf, ff):
            loss += F.l1_loss(f, r)
    return loss * 2

def discriminator_loss(real_outputs, fake_outputs):
    loss = 0
    for real, fake in zip(real_outputs, fake_outputs):
        loss += torch.mean((1 - real) ** 2) + torch.mean(fake ** 2)
    return loss

def generator_loss(fake_outputs):
    loss = 0
    for fake in fake_outputs:
        loss += torch.mean((1 - fake) ** 2)
    return loss
