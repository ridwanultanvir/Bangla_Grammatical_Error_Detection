from src.utils.mapper import configmapper
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

import pdb

from normalizer import normalize


@configmapper.map("datasets", "toxic_spans_crf_4cls_tokens")
class ToxicSpansCRF4ClsTokenDataset:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_checkpoint_name
        )
        self.dataset = load_dataset("csv", data_files=dict(self.config.train_files))
        self.test_dataset = load_dataset("csv", data_files=dict(self.config.eval_files))

        self.tokenized_inputs = self.dataset.map(
            self.tokenize_and_align_labels_for_train, batched=True
        )
        self.test_tokenized_inputs = self.test_dataset.map(
            self.tokenize_for_test, batched=True
        )

    def tokenize_and_align_labels_for_train(self, examples):
        # print("Tokenizing and aligning labels for train...")
        # examples["text"] = normalize(examples["text"])
        tokenized_inputs = self.tokenizer(
            examples["text"], **self.config.tokenizer_params
        )

        # tokenized_inputs["text"] = examples["text"]
        example_spans = []
        labels = []
        prediction_mask = np.zeros_like(np.array(tokenized_inputs["input_ids"]))
        offsets_mapping = tokenized_inputs["offset_mapping"]

        ## Wrong Code
        # for i, offset_mapping in enumerate(offsets_mapping):
        #     j = 0
        #     while j < len(offset_mapping):  # [tok1, tok2, tok3] [(0,5),(1,4),(5,7)]
        #         if tokenized_inputs["input_ids"][i][j] in [
        #             self.tokenizer.sep_token_id,
        #             self.tokenizer.pad_token_id,
        #             self.tokenizer.cls_token_id,
        #         ]:
        #             j = j + 1
        #             continue
        #         else:
        #             k = j + 1
        #             while self.tokenizer.convert_ids_to_tokens(
        #                 tokenized_inputs["input_ids"][i][k]
        #             ).startswith("##"):
        #                 offset_mapping[i][j][1] = offset_mapping[i][k][1]
        #             j = k

        for i, offset_mapping in enumerate(offsets_mapping):
            labels.append([])

            spans = eval(examples["spans"][i])
            Bs = eval(examples["Bs"][i])
            Is = eval(examples["Is"][i])
            Es = eval(examples["Es"][i])

            example_spans.append(spans)
            # cls_label = 2  ## DUMMY LABEL
            cls_label = 4  ## DUMMY LABEL
            for j, offsets in enumerate(offset_mapping):
                if tokenized_inputs["input_ids"][i][j] in [
                    self.tokenizer.sep_token_id,
                    self.tokenizer.pad_token_id,
                ]:
                    tokenized_inputs["attention_mask"][i][j] = 0

                if tokenized_inputs["input_ids"][i][j] == self.tokenizer.cls_token_id:
                    labels[-1].append(cls_label)
                    prediction_mask[i][j] = 1

                elif offsets[0] == offsets[1] and offsets[0] == 0:
                    # labels[-1].append(2)  ## DUMMY
                    labels[-1].append(cls_label)  ## DUMMY

                else:
                    # toxic_offsets = [x in spans for x in range(offsets[0], offsets[1])]
                    # ## If any part of the the token is in span, mark it as Toxic
                    # if (
                    #     len(toxic_offsets) > 0
                    #     and sum(toxic_offsets) / len(toxic_offsets) > 0.0
                    # ):
                    #     labels[-1].append(1)
                    # else:
                    #     labels[-1].append(0)

                    prediction_mask[i][j] = 1
                    
                    b_off = [x in Bs for x in range(offsets[0], offsets[1])]
                    b_off = sum(b_off)
                    i_off = [x in Is for x in range(offsets[0], offsets[1])]
                    i_off = sum(i_off)
                    e_off = [x in Es for x in range(offsets[0], offsets[1])]
                    e_off = sum(e_off)

                    # if len(b_off) == len(i_off) and len(i_off)  == 0:
                    if b_off == 0 and i_off == 0 and e_off == 0:
                        labels[-1].append(0)
                    # elif len(b_off) >= len(i_off) == 1:
                    elif b_off >= i_off and b_off >= e_off:
                        labels[-1].append(1)
                    elif i_off >= b_off and i_off >= e_off:
                        labels[-1].append(2)
                    elif e_off >= b_off and e_off >= i_off:
                        labels[-1].append(3)
        
        # pdb.set_trace()

        tokenized_inputs["labels"] = labels
        tokenized_inputs["prediction_mask"] = prediction_mask
        return tokenized_inputs

    def tokenize_for_test(self, examples):
        # examples["text"] = normalize(examples["text"])
        tokenized_inputs = self.tokenizer(
            examples["text"], **self.config.tokenizer_params
        )
        prediction_mask = np.zeros_like(np.array(tokenized_inputs["input_ids"]))
        labels = np.zeros_like(np.array(tokenized_inputs["input_ids"]))
        
        offsets_mapping = tokenized_inputs["offset_mapping"]

        for i, offset_mapping in enumerate(offsets_mapping):
            for j, offsets in enumerate(offset_mapping):
                if tokenized_inputs["input_ids"][i][j] in [
                    self.tokenizer.sep_token_id,
                    self.tokenizer.pad_token_id,
                ]:
                    tokenized_inputs["attention_mask"][i][j] = 0
                else:
                    prediction_mask[i][j] = 1
        
        tokenized_inputs["prediction_mask"] = prediction_mask
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
