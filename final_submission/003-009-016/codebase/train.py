"""Train File."""
## Imports
import argparse

# import itertools
import copy
import os
import numpy as np
from omegaconf import OmegaConf

import torch
import torch.nn as nn

from copy import deepcopy
from datasets import load_metric
from evaluation.semeval2021 import f1
from sklearn.metrics import f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorForTokenClassification,
    default_data_collator,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import f1_score
from src.utils.configuration import Config

from src.datasets import *
from src.models import *

from src.modules.preprocessors import *
from src.utils.mapper import configmapper

import os
import gc
import ipdb

import sys

untokenized_train_dataset_cache = None
# TODO: do for three classes
def compute_metrics_token(p):
    # print(type(p))
    # print(dir(p))
    # print(p.inputs)
    predictions, labels = p
    # import pdb; pdb.set_trace()

    multihead = False
    if len(labels.shape) == 3:
        multihead = True
    
        
    # print(predictions.shape, labels.shape)
    # pdb.set_trace()
    if multihead:
        labels_0 = labels[..., 0]
        labels_1 = labels[..., 1]
        predictions_0 = predictions[..., :3]
        predictions_1 = predictions[..., 3:]
        predictions_0 = np.argmax(predictions_0, axis=2)
        predictions_1 = np.argmax(predictions_1, axis=2)
    else:        
        predictions = np.argmax(predictions, axis=2)  ## batch_size, seq_length

    offset_wise_scores = []
    # print(len(predictions))
    # os.system('spd-say "Tracing begins"'); ipdb.set_trace()
    '''
    [i for i, pred in enumerate(predictions) if 2 in pred]
    '''

    if multihead:
        for i, prediction in enumerate(predictions_0):
            ## Batch Wise
            # print(len(prediction))
            # print("Prediction: ", prediction); exit()
            # pdb.set_trace()
            ground_spans = eval(validation_spans_0[i])
            predicted_spans = []
            # ipdb.set_trace()
            for j, tokenwise_prediction in enumerate(
                prediction[: len(validation_offsets_mapping[i])]
            ):
                # pdb.set_trace()
                # if tokenwise_prediction == 1:
                # if tokenwise_prediction == in [1, 2]:
                if tokenwise_prediction in range(1, train_config.pretrained_args.num_labels):
                    predicted_spans += list(
                        range(
                            validation_offsets_mapping[i][j][0],
                            validation_offsets_mapping[i][j][1],
                        )
                    )
            f1_val = f1(predicted_spans, ground_spans)     
            # print("f1 B-I: ", f1_val)           
            offset_wise_scores.append(f1_val)
        print("Prediction: ", predictions_1.sum())
        for i, prediction in enumerate(predictions_1):
            ## Batch Wise
            # print(len(prediction))
            # print("Prediction: ", prediction); exit()
            # pdb.set_trace()
            ground_spans = eval(validation_spans_1[i])
            predicted_spans = []
            # ipdb.set_trace()
            for j, tokenwise_prediction in enumerate(
                prediction[: len(validation_offsets_mapping[i])]
            ):
                # pdb.set_trace()
                if tokenwise_prediction == 1:

                # if tokenwise_prediction == in [1, 2]:
                # if tokenwise_prediction in range(1, train_config.pretrained_args.num_labels):
                    predicted_spans += list(
                        range(
                            validation_offsets_mapping[i][j][0],
                            validation_offsets_mapping[i][j][1],
                        )
                    )
                
            f1_val = f1(predicted_spans, ground_spans)       
            # print("f1 E: ", f1_val)         
            offset_wise_scores.append(f1_val)
        
        results_offset = np.mean(offset_wise_scores)
        # pdb.set_trace()

        true_predictions = [
            [p for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions_0, labels_0)
        ] 
        true_predictions_1 =  [
            [p for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions_1, labels_1)
        ]
        true_labels = [
            [l for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions_0, labels_0)
        ] 
        true_labels_1 = [
            [l for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions_1, labels_1)
        ]

        
        results = np.mean(
            [
                f1_score(true_label, true_preds) if train_config.pretrained_args.num_labels == 2 
                    # else f1_score(true_label, true_preds, average="macro")
                    else f1_score(true_label, true_preds, average="weighted")

                for true_label, true_preds in zip(true_labels, true_predictions)
            ] +  [
                f1_score(true_label, true_preds) 
                    # if train_config.pretrained_args.num_labels == 2 
                    # else f1_score(true_label, true_preds, average="macro")
                    # else f1_score(true_label, true_preds, average="weighted")

                for true_label, true_preds in zip(true_labels_1, true_predictions_1)
            ]
        )
        return {"Token-Wise F1": results, "Offset-Wise F1": results_offset}
    else:
    
        for i, prediction in enumerate(predictions):
            ## Batch Wise
            # print(len(prediction))
            # print("Prediction: ", prediction); exit()
            # pdb.set_trace()
            ground_spans = eval(validation_spans[i])
            predicted_spans = []
            # ipdb.set_trace()
            for j, tokenwise_prediction in enumerate(
                prediction[: len(validation_offsets_mapping[i])]
            ):
                # pdb.set_trace()
                # if tokenwise_prediction == 1:
                # if tokenwise_prediction == in [1, 2]:
                if tokenwise_prediction in range(1, train_config.pretrained_args.num_labels):
                    predicted_spans += list(
                        range(
                            validation_offsets_mapping[i][j][0],
                            validation_offsets_mapping[i][j][1],
                        )
                    )
                
            offset_wise_scores.append(f1(predicted_spans, ground_spans))
        results_offset = np.mean(offset_wise_scores)

        true_predictions = [
            [p for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [l for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]

        
        results = np.mean(
            [
                f1_score(true_label, true_preds) if train_config.pretrained_args.num_labels == 2 
                    # else f1_score(true_label, true_preds, average="macro")
                    else f1_score(true_label, true_preds, average="weighted")

                for true_label, true_preds in zip(true_labels, true_predictions)
            ]
        )
        return {"Token-Wise F1": results, "Offset-Wise F1": results_offset}


dirname = os.path.dirname(__file__)  ## For Paths Relative to Current File

## Config
parser = argparse.ArgumentParser(prog="train.py", description="Train a model.")
parser.add_argument(
    "--train",
    type=str,
    action="store",
    help="The configuration for model training/evaluation",
)
parser.add_argument(
    "--data",
    type=str,
    action="store",
    help="The configuration for data",
)

args = parser.parse_args()
# print(vars(args))
train_config = OmegaConf.load(args.train)
# sys.stdout = open(f'{train_config.args.output_dir}/out.log', 'a+')
# Redirect stdout to file, if file exists, append to it else create a new file


data_config = OmegaConf.load(args.data)

print(data_config.train_files)
dataset = configmapper.get_object("datasets", data_config.name)(data_config)
untokenized_train_dataset = dataset.dataset
tokenized_train_dataset = dataset.tokenized_inputs
tokenized_test_dataset = dataset.test_tokenized_inputs


model_class = configmapper.get_object("models", train_config.model_name)

if "toxic-bert" in train_config.pretrained_args.pretrained_model_name_or_path:
    toxicbert_model = AutoModelForSequenceClassification.from_pretrained(
        train_config.pretrained_args.pretrained_model_name_or_path
    )
    train_config.pretrained_args.pretrained_model_name_or_path = "bert-base-uncased"
    model = model_class.from_pretrained(**train_config.pretrained_args)
    model.bert = deepcopy(toxicbert_model.bert)
    gc.collect()

elif "toxic-roberta" in train_config.pretrained_args.pretrained_model_name_or_path:
    toxicroberta_model = AutoModelForSequenceClassification.from_pretrained(
        train_config.pretrained_args.pretrained_model_name_or_path
    )
    train_config.pretrained_args.pretrained_model_name_or_path = "roberta-base"
    model = model_class.from_pretrained(**train_config.pretrained_args)
    model.roberta = deepcopy(toxicroberta_model.roberta)
    gc.collect()

else:
    model = model_class.from_pretrained(**train_config.pretrained_args)
    # import pdb; pdb.set_trace()
    if 'freeze_backbone' in train_config.keys() and train_config.freeze_backbone:
        for param in model.electra.parameters():
            param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained(data_config.model_checkpoint_name)
# import pdb; pdb.set_trace()
# wordlist = '../dcspell/data/two_dictwords.csv'
# df = pd.read_csv(wordlist)
# words = df['word'].tolist()

# from tqdm import tqdm

# for word in tqdm(words):
#     tokens = tokenizer.tokenize(word)
#     if len(tokens) == 1 and tokens[0] == '[UNK]' and word != '[UNK]':
#         tokenizer.add_tokens(word)

# model.resize_token_embeddings(len(tokenizer))

# if data_config.model_checkpoint_name == "sberbank-ai/mGPT":
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

if "crf" in train_config.model_name:
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = None
    # compute_metrics = compute_metrics_token
elif not "spans" in train_config.model_name:
    validation_spans = untokenized_train_dataset["validation"]["spans"]
    if train_config.multihead:
        validation_spans_0 = (
            untokenized_train_dataset["validation"]["Bs"] + 
            untokenized_train_dataset["validation"]["Is"]
        )
        validation_spans_1 = untokenized_train_dataset["validation"]["Es"]
    
    validation_offsets_mapping = tokenized_train_dataset["validation"]["offset_mapping"]
    data_collator = DataCollatorForTokenClassification(tokenizer)
    compute_metrics = compute_metrics_token

else:
    data_collator = default_data_collator
    compute_metrics = None

## Need to place data_collator
if "multi" in train_config.model_name:
    args = TrainingArguments(
        label_names=["start_positions", "end_positions"], **train_config.args
    )
else:
    args = TrainingArguments(**train_config.args)

if not os.path.exists(train_config.args.output_dir):
    os.makedirs(train_config.args.output_dir)

sys.stdout = open(f'{train_config.args.output_dir[:-7]}/out.log', 'a+')
checkpoints = sorted(
    os.listdir(train_config.args.output_dir), key=lambda x: int(x.split("-")[1])
)
if len(checkpoints) != 0:
    print("Found Checkpoints:")
    print(checkpoints)

# print(untokenized_train_dataset["validation"])
# exit()
# global untokenized_train_dataset_cache
untokenized_train_dataset_cache = untokenized_train_dataset
# Find class weights for the dataset

from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

class CustomTrainer(Trainer):
    # Pass all to super
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        loss_weights = [1.0]
        error_weight = 10.0 # not working well
        for i in range(1, self.model.config.num_labels):
            loss_weights.append(error_weight)
        self.loss_weights = torch.tensor(loss_weights).cuda()

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # https://huggingface.co/docs/transformers/main_classes/trainer
    #     labels = inputs.get("labels")
    #     # forward pass
    #     outputs = model(**inputs)
    #     logits = outputs.get("logits")
    #     # compute custom loss (suppose one has 3 labels with different weights)

    #     # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0, 10.0]))
    #     loss_fct = nn.CrossEntropyLoss(weight=self.loss_weights)
    #     loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #     return (loss, outputs) if return_outputs else loss
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        multihead = False
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
            if len(labels.shape) == 3:
                multihead = True
                labels_0 = labels[:, :, 0]
                labels_1 = labels[:, :, 1]
                labels_0 = labels_0.squeeze(-1)
                labels_1 = labels_1.squeeze(-1)
        else:
            labels = None
        outputs = model(**inputs)
        if multihead:
            # outputs = outputs[0]
            # pdb.set_trace()
            # outputs_0 = outputs.a
            # outputs_1 = outputs.b
            
            outputs_0 = outputs.logits[..., :3]
            outputs_1 = outputs.logits[..., 3:]
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # pdb.set_trace()
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                if multihead:
                    loss =  self.label_smoother([outputs_0], labels_0, shift_labels=True)
                    loss += self.label_smoother([outputs_1], labels_1, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                if multihead:
                    loss =  self.label_smoother([outputs_0], labels_0)
                    loss += self.label_smoother([outputs_1], labels_1)
                else:
                    loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            # if multihead:
            #     loss = outputs[0]["loss"] if isinstance(outputs[0], dict) else outputs[0][0]
            # else:
            if 1:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # return (loss, outputs) if return_outputs else loss
        # if multihead:
        #     return (loss, {"outputs": outputs}) if return_outputs else loss
        # else:
        if 1:
            return (loss, outputs) if return_outputs else loss


# trainer = CustomTrainer(
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset["train"],
    eval_dataset=tokenized_train_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

if len(checkpoints) != 0:
    trainer.train(
        os.path.join(train_config.args.output_dir, checkpoints[-1])
    )  ## Load from checkpoint
else:
    trainer.train()
if not os.path.exists(train_config.save_model_path):
    os.makedirs(train_config.save_model_path)
trainer.save_model(train_config.save_model_path)

os.system('spd-say "Training is done."')
