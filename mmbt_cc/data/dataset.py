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


import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from data.category_mapping import get_category_mapping


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]
        self.category_mapping = get_category_mapping(args)

        self.max_seq_len = args.max_seq_len
        if args.model == "mmbt":
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = (
                self.text_start_token
                + self.tokenizer(self.data[index]["caliber"])[
                  : (self.args.max_seq_len - 1)
                  ]
        )
        segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        label = torch.LongTensor(
            [self.args.labels.index(self.data[index]["category"]) if self.data[index]["category"]
                in self.category_mapping else self.args.labels.index("other")]
        )

        image = None
        if self.args.model in ["img", "mmbt"]:
            if self.data[index]["image_file"]:
                image = Image.open(
                    os.path.join(self.data_dir, 'images/', self.data[index]["image_file"])
                ).convert("RGB")
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)

        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1

        return sentence, segment, image, label
