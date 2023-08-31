#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE_MMBT file in the Licenses directory of this source tree.
#

# Adaptations to the original file are made by Sarah de Boer.
# The adaptations are licensed under the LICENSE file in the root directory.


from models.image import ImageClf
from models.mmbt import MultimodalBertClf


MODELS = {
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
}


def get_model(args):
    return MODELS[args.model](args)
