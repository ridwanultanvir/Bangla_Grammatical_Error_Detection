import torch
# from transformers import BertForTokenClassification
from transformers import ElectraForTokenClassification
from torchcrf import CRF
from src.utils.mapper import configmapper
# import pdb


@configmapper.map("models", "bert_crf_3cls_token")
# class BertLSTMCRF(BertForTokenClassification):
class BertLSTMCRF(ElectraForTokenClassification):
    def __init__(self, config, lstm_hidden_size, lstm_layers):
        print("bert_crf_3cls_token")
        super().__init__(config)
        # ipdb.set_trace()
        self.lstm = torch.nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            dropout=0.2,
            batch_first=True,
            bidirectional=True,
        )
        # print("num_labels: ", config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        del self.classifier
        self.classifier = torch.nn.Linear(2 * lstm_hidden_size, config.num_labels)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        prediction_mask=None,
    ):
        # pdb.set_trace()

        # outputs = self.bert(
        outputs = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            output_hidden_states=True,
            return_dict=False,
        )
        # seq_output, all_hidden_states, all_self_attntions, all_cross_attentions

        sequence_output = outputs[0]  # outputs[1] is pooled output which is none.

        sequence_output = self.dropout(sequence_output)

        lstm_out, *_ = self.lstm(sequence_output)
        sequence_output = self.dropout(lstm_out)

        logits = self.classifier(sequence_output)

        ## CRF
        # print(logits.size())
        mask = prediction_mask
        mask = mask[:, : logits.size(1)].contiguous()

        # print(logits)

        if labels is not None:
            labels = labels[:, : logits.size(1)].contiguous()
            loss = -self.crf(logits, labels, mask=mask.bool(), reduction="token_mean")

        tags = self.crf.decode(logits, mask.bool())
        # print(tags)
        if labels is not None:
            return (loss, logits, tags)
        else:
            return (logits, tags)
