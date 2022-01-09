# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Grzegorz Jacenków.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Transformers' MMBT model. We had to make some changes to the model.

Taken from: transformers/models/mmbt/modeling_mmbt.py
 """


import torch
from torch import nn
from transformers.modeling_utils import ModuleUtilsMixin


class MMBTConfig:
    def __init__(self, config, num_labels=None, modal_hidden_size=2048):
        self.__dict__ = config.__dict__
        self.modal_hidden_size = modal_hidden_size
        self.use_return_dict = False
        if num_labels:
            self.num_labels = num_labels


class ModalEmbeddings(nn.Module):
    """Modal Embeddings which takes an encoder, and a transformer embedding."""
    def __init__(self, config, encoder, embeddings):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.proj_embeddings = nn.Linear(
            config.modal_hidden_size, config.hidden_size)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def forward(self, input_modal, start_token=None, end_token=None,
                position_ids=None, token_type_ids=None):
        token_embeddings = self.proj_embeddings(self.encoder(input_modal))
        seq_length = token_embeddings.size(1)

        if start_token is not None:
            start_token_embeds = self.word_embeddings(start_token)
            seq_length += 1
            token_embeddings = torch.cat(
                [start_token_embeds.unsqueeze(1), token_embeddings], dim=1)

        if end_token is not None:
            end_token_embeds = self.word_embeddings(end_token)
            seq_length += 1
            token_embeddings = torch.cat(
                [token_embeddings, end_token_embeds.unsqueeze(1)], dim=1)

        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_modal.device)
            position_ids = position_ids.unsqueeze(0).expand(
                input_modal.size(0), seq_length)

        if token_type_ids is None:
            token_type_ids = torch.zeros(
                (input_modal.size(0), seq_length),
                dtype=torch.long, device=input_modal.device
            )

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings =\
            token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class MMBTModel(nn.Module, ModuleUtilsMixin):
    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.modal_encoder = ModalEmbeddings(
            config, encoder, transformer.embeddings)

    def forward(
        self,
        input_modal,
        input_ids=None,
        modal_start_tokens=None,
        modal_end_tokens=None,
        attention_mask=None,
        token_type_ids=None,
        modal_token_type_ids=None,
        position_ids=None,
        modal_position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None \
            else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and ",
                             "inputs_embeds at the same time")
        elif input_ids is not None:
            input_txt_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_txt_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None \
            else inputs_embeds.device

        modal_embeddings = self.modal_encoder(
            input_modal,
            start_token=modal_start_tokens,
            end_token=modal_end_tokens,
            position_ids=modal_position_ids,
            token_type_ids=modal_token_type_ids,
        )

        input_modal_shape = modal_embeddings.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.ones(
                input_txt_shape, dtype=torch.long, device=device)

        txt_embeddings = self.transformer.embeddings(
            input_ids=input_ids, position_ids=position_ids,
            token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        embedding_output = torch.cat([modal_embeddings, txt_embeddings], 1)
        input_shape = embedding_output.size()[:-1]

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        else:
            attention_mask = torch.cat(
                [torch.ones(input_modal_shape, device=device, dtype=torch.long),
                 attention_mask], dim=1
            )
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        else:
            encoder_attention_mask = torch.cat(
                [torch.ones(input_modal_shape, device=device),
                 encoder_attention_mask], dim=1
            )

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, self.device)
        encoder_extended_attention_mask = self.invert_attention_mask(
            encoder_attention_mask)
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.transformer.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.transformer.pooler(sequence_output)

        return sequence_output, pooled_output

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
