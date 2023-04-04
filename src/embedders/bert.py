# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch.utils.checkpoint
import torch.nn as nn
import torch

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.utils import logging


class SemanticEmbeddings(nn.Module):
    """Construct the embeddings from word only."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """generate embeddings for the input_ids"""
        inputs_embeds = self.word_embeddings(input_ids)
        return inputs_embeds


class BertForEmbedding(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"predictions.decoder.bias",
        r"cls.predictions.decoder.weight",
    ]

    def __init__(self, config):
        super().__init__(config)

        # create embeddings
        self.embeddings = SemanticEmbeddings(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self, input_ids: torch.Tensor, *args, **kwargs  # absorb extra arguments
    ) -> torch.Tensor:
        """generate embeddings for the input_ids"""
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        embedding_output = self.embeddings(input_ids=input_ids)

        return embedding_output
