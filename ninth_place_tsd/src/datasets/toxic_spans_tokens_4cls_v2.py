from src.utils.mapper import configmapper
from transformers import AutoTokenizer
from datasets import load_dataset

import pdb

@configmapper.map("datasets", "toxic_spans_tokens_4cls_v2")
class ToxicSpansToken4CLSDatasetV2:
    def __init__(self, config):
        # print("### ToxicSpansTokenDataset ###"); exit()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_checkpoint_name
        )
        # if self.config.model_checkpoint_name == "sberbank-ai/mGPT":
            # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.dataset = load_dataset("csv", data_files=dict(self.config.train_files))
        self.test_dataset = load_dataset("csv", data_files=dict(self.config.eval_files))

        self.tokenized_inputs = self.dataset.map(
            self.tokenize_and_align_labels_for_train, batched=True
        )
        self.test_tokenized_inputs = self.test_dataset.map(
            self.tokenize_for_test, batched=True
        )

    def tokenize_and_align_labels_for_train(self, examples):
        # print("4 CLS")

        tokenized_inputs = self.tokenizer(
            examples["text"], **self.config.tokenizer_params
        )

        # tokenized_inputs["text"] = examples["text"]
        example_spans = []
        labels = []
        # labels_E = []
    
        offsets_mapping = tokenized_inputs["offset_mapping"]
        # pdb.set_trace()
        for i, offset_mapping in enumerate(offsets_mapping):
            labels.append([])
            # labels_E.append([])

            spans = eval(examples["spans"][i])
            Bs = eval(examples["Bs"][i])
            Is = eval(examples["Is"][i])
            Es = eval(examples["Es"][i])
            example_spans.append(spans)
            if self.config.label_cls:
                cls_label = (
                    1
                    if (
                        len(examples["text"][i]) > 0
                        and len(spans) / len(examples["text"][i])
                        > self.config.cls_threshold
                    )
                    else 0
                )  ## Make class label based on threshold
            else:
                cls_label = -100
            for j, offsets in enumerate(offset_mapping):
                if tokenized_inputs["input_ids"][i][j] == self.tokenizer.cls_token_id:
                    # labels[-1].append(cls_label)
                    # labels_E[-1].append(cls_label)
                    label = label_E = cls_label
                elif offsets[0] == offsets[1] and offsets[0] == 0: # All zero
                    # labels[-1].append(-100)  ## SPECIAL TOKEN
                    # labels_E[-1].append(-100)  ## SPECIAL TOKEN
                    label = label_E = -100
                else:
                    # toxic_offsets = [x in spans for x in range(offsets[0], offsets[1])]
                    ## If any part of the the token is in span, mark it as Toxic
                    # if (
                    #     len(toxic_offsets) > 0
                    #     and sum(toxic_offsets) / len(toxic_offsets)
                    #     > self.config.token_threshold
                    # ):
                    #     labels[-1].append(1)
                    # else:
                    #     labels[-1].append(0)
                    b_off = [x in Bs for x in range(offsets[0], offsets[1])]
                    b_off = sum(b_off)
                    i_off = [x in Is for x in range(offsets[0], offsets[1])]
                    i_off = sum(i_off)
                    e_off = [x in Es for x in range(offsets[0], offsets[1])]
                    e_off = sum(e_off)

                    if b_off == 0 and i_off == 0:
                        # labels[-1].append(0)
                        label = 0
                    # elif len(b_off) >= len(i_off) == 1:
                    elif b_off >= i_off:
                        # labels[-1].append(1)
                        label = 1
                        # print(b_off)
                        # print(i_off)
                        # print(j)
                    else:
                        # labels[-1].append(2)
                        label = 2

                    if e_off == 0:
                        # labels_E[-1].append(0)
                        label_E = 0
                    else:
                        # labels_E[-1].append(1)
                        # print("e_off", label_E)
                        label_E = 1

                    # if len(b_off) == len(i_off) and len(i_off)  == 0:
                    # if b_off == 0 and i_off == 0 and e_off == 0:
                    #     labels[-1].append(0)
                    # # elif len(b_off) >= len(i_off) == 1:
                    # elif b_off >= i_off and b_off >= e_off:
                    #     labels[-1].append(1)
                    # elif i_off >= b_off and i_off >= e_off:
                    #     labels[-1].append(2)
                    # elif e_off >= b_off and e_off >= i_off:
                    #     labels[-1].append(3)
                    # elif b_off >= i_off:
                    #     labels[-1].append(1)
                    #     # print(b_off)
                    #     # print(i_off)
                    #     # print(j)
                    # else:
                    #     labels[-1].append(2)
                
                labels[-1].append((label, label_E))

            # pdb.set_trace()



        tokenized_inputs["labels"] = labels
        # pdb.set_trace()
        # print("tokenized_inputs", tokenized_inputs); exit()
        return tokenized_inputs

    def tokenize_for_test(self, examples):
        # print("4 CLS")
        tokenized_inputs = self.tokenizer(
            examples["text"], **self.config.tokenizer_params
        )
        return tokenized_inputs
