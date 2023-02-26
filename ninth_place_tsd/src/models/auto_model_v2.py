import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss


from transformers import ElectraModel, ElectraPreTrainedModel
from transformers.modeling_outputs import (
  TokenClassifierOutput
)

from src.utils.mapper import configmapper
# import pdb


class Dummy:
  def __init__(self, a, b):
    self.a = a
    self.b = b  


@configmapper.map("models", "autotoken_4cls_v2")
class ElectraForTokenClassification_v2(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.electra = ElectraModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_2 = nn.Linear(config.hidden_size, 2)
        # Initialize weights and apply final processing
        self.post_init()

    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]

        discriminator_sequence_output = self.dropout(discriminator_sequence_output)
        logits = self.classifier(discriminator_sequence_output)
        logits_2 = self.classifier_2(discriminator_sequence_output)

        loss = loss_1 = None
        if labels is not None:
            labels_0 = labels[..., 0]
            labels_1 = labels[..., 1]
            # import pdb; pdb.set_trace()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels_0.view(-1))
            loss_1 = loss_fct(logits_2.view(-1, 2), labels_1.view(-1))

        if not return_dict:
            output = (logits,) + discriminator_hidden_states[1:]
            return ((loss,) + output) if loss is not None else output
        
        # import pdb; pdb.set_trace()
        # Stack two logits of unequal size

        return TokenClassifierOutput(
            loss=torch.stack([loss, loss_1]) if loss is not None else None,
            logits=torch.cat([logits, logits_2], axis=-1),
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )

        # return [TokenClassifierOutput(
        #     loss=loss,
        #     logits=logits,
        #     hidden_states=discriminator_hidden_states.hidden_states,
        #     attentions=discriminator_hidden_states.attentions,
        # ), TokenClassifierOutput(
        #     loss=loss_1,
        #     logits=logits_2,
        #     hidden_states=discriminator_hidden_states.hidden_states,
        #     attentions=discriminator_hidden_states.attentions,
        # )]