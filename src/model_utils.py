# src/model_utils.py

import torch.nn as nn
import torchvision.models as models

def create_vgg16(fine_tune_last_block: bool = False):
    model = models.vgg16(pretrained=True)
    # freeze all
    for p in model.parameters():
        p.requires_grad = False

    if fine_tune_last_block:
        # unfreeze features[-4:] → last conv block
        for p in model.features[-4:].parameters():
            p.requires_grad = True

    # replace classifier
    in_feats = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_feats, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(256, 1)   # single logit
    )
    return model
