"""Eval File."""
## Imports
import argparse

# import itertools
import copy
import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from evaluation.semeval2021 import f1
from sklearn.metrics import f1_score

from transformers import (
  AutoTokenizer,
  DataCollatorForTokenClassification,
  default_data_collator,
  pipeline,
  Trainer,
  TrainingArguments,
)
from sklearn.metrics import f1_score
from src.utils.configuration import Config

from src.datasets import *
from src.models import *

from src.modules.preprocessors import *
from src.utils.mapper import configmapper
from src.utils.postprocess_predictions import (
  postprocess_token_span_predictions,
  postprocess_multi_span_predictions,
)

from tqdm.auto import tqdm
import os


def compute_metrics_token(p):
  # print(type(p)); exit()
  predictions, labels = p
  predictions = np.argmax(predictions, axis=2)  ## batch_size, seq_length

  offset_wise_scores = []
  # print(len(predictions))
  for i, prediction in enumerate(predictions):
    ## Batch Wise
    # print(len(prediction))
    ground_spans = eval(validation_spans[i])
    predicted_spans = []
    for j, tokenwise_prediction in enumerate(
      prediction[: len(validation_offsets_mapping[i])]
    ):
      if tokenwise_prediction == 1:
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
      f1_score(true_label, true_preds)
      for true_label, true_preds in zip(true_labels, true_predictions)
    ]
  )
  return {"Token-Wise F1": results, "Offset-Wise F1": results_offset}


def predict_tokens_spans(model, dataset, examples, tokenizer):
  trainer = Trainer(
    model,
  )
  raw_predictions = trainer.predict(dataset)
  dataset.set_format(
    type=dataset.format["type"], columns=list(dataset.features.keys())
  )
  final_predictions = postprocess_token_span_predictions(
    dataset, examples, raw_predictions.predictions, tokenizer
  )
  return final_predictions


def get_token_spans_separate_logits(model, dataset, type="spans"):
  trainer = Trainer(
    model,
  )
  raw_predictions = trainer.predict(dataset)
  start_logits, end_logits, token_logits = raw_predictions.predictions
  dataset.set_format(
    type=dataset.format["type"], columns=list(dataset.features.keys())
  )
  if type == "spans":
    return start_logits, end_logits
  else:
    return token_logits


def predict_multi_spans(model, dataset, examples, tokenizer):
  trainer = Trainer(
    model,
  )
  raw_predictions = trainer.predict(dataset)
  dataset.set_format(
    type=dataset.format["type"], columns=list(dataset.features.keys())
  )
  final_predictions = postprocess_multi_span_predictions(
    dataset, examples, raw_predictions.predictions, tokenizer
  )
  return final_predictions


dirname = os.path.dirname(__file__)  ## For Paths Relative to Current File

## Config
parser = argparse.ArgumentParser(prog="eval.py", description="Evaluate a model.")
parser.add_argument(
  "--eval",
  type=str,
  action="store",
  help="The configuration for model training/evaluation",
)

# Add an arg model_checkpoint_name
parser.add_argument(
  "--model_checkpoint_name",
  type=str,
  action="store",
  help="The model checkpoint name",
  default=None
)



args = parser.parse_args()
# print(vars(args))
eval_config = OmegaConf.load(args.eval)
data_config = eval_config.dataset
if args.model_checkpoint_name is not None:
  data_config.model_checkpoint_name = eval_config.results_dir + "/ckpts/" + args.model_checkpoint_name

## Scope for Improvement:
# 1. Remove all but one train files in the eval config, move all to test, will be easier to load and process during the prediction.
# 2. Add an option to exclude train/test files.
# 3. Remove redundant code from the eval script.
# 4. Make all postprocessing functions same with more features.
# 5. Use fn in predict_xyz, and pass function based on type.
dataset = configmapper.get_object("datasets", data_config.name)(data_config)
untokenized_train_dataset = dataset.dataset
untokenized_test_dataset = dataset.test_dataset
tokenized_train_dataset = dataset.tokenized_inputs
tokenized_test_dataset = dataset.test_tokenized_inputs

model_class = configmapper.get_object("models", eval_config.model_name)
model = model_class.from_pretrained(**eval_config.pretrained_args)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(data_config.model_checkpoint_name)
suffix = data_config.model_checkpoint_name.split("/")[-1]

from operator import itemgetter
from itertools import groupby

def get_ranges(data):
  ranges = []
  for key, group in groupby(enumerate(data), lambda x:x[0]-x[1]):
    group = list(map(itemgetter(1), group))
    if len(group) > 1:
      ranges.append(range(group[0], group[-1]))
    else:
    #   ranges.append(group[0])
      ranges.append(range(group[0], group[0]))
  return ranges


space_chars = ['।', '?', '!', ","]
def replace_fixed(x):
  for char in space_chars:
    x = x.replace(" "+char, "$ $"+char)
  
  return x

end_chars = ['।', '?', '!', 
          # '$'
]
end_char_before = ['।', '?', '!', '.', '*']
end_before = [x+ "$" for x in end_char_before]
# temp = open("temp.txt", "w")
def replace_end(x):
  if len(x) == 0:
    return x
  # Find the last character position from the end of the string which is not a space
  # last_char_pos = len(x) - 1
  # while x[last_char_pos] == " ":
  #   last_char_pos -= 1
  # Find the last character position from the end of the string which is not a space using regex
  # Already handled
  # if x.endswith("$$$"):
  #   print(x)
  # temp.write(x + "\n")


  if x.endswith("$$$$"):
    # print("End with $$$$", x)
    return x[:-2]

  if len(x) >= 2 and x[-2:] in end_before:
    # print("End with punct$", x)
    # print(x)
    return x

  if x.endswith("$$"):
    return x

  # last_char_pos = re.search(r"\s+$", x)
  # if last_char_pos is None:
  #   return x if x[-1] in end_chars else x + "$$"
  # last_char_pos = last_char_pos.start()
  # # If the last character is not in end_chars then add a  $$
  # return x if x[last_char_pos] in end_chars else x[:last_char_pos+1] + "$$" + x[last_char_pos+1:]
  x =  x if x[-1] in end_chars else x + "$$"

  if x.endswith('$$ $$'):
    # print("End with $$ $$", x)
    x = x[:-5] + '$$ '
  
  return x

if "crf" in eval_config.model_name:
  data_collator = DataCollatorForTokenClassification(tokenizer)
  model = model.cuda()
elif "token_spans" in eval_config.model_name or "multi" in eval_config.model_name:
  data_collator = default_data_collator

elif "token" in eval_config.model_name:
  validation_spans = untokenized_train_dataset["validation"]["spans"]
  validation_offsets_mapping = tokenized_train_dataset["validation"]["offset_mapping"]
  data_collator = DataCollatorForTokenClassification(tokenizer)
  compute_metrics = compute_metrics_token

else:
  nlp = pipeline(task="question-answering", model=model, tokenizer=tokenizer)
  data_collator = default_data_collator

## Need to place data_collator
# from nltk.metrics import edit_distance
import Levenshtein
edit_distance = Levenshtein.distance


if not os.path.exists(eval_config.save_dir):
  os.makedirs(eval_config.save_dir)

print("Eval_config.model_name: ", eval_config.model_name)

from transformers import Trainer
from transformers.trainer import (
  EvalLoopOutput, DataLoader,
  deepspeed_init,
  logger,
  is_torch_tpu_available,
  # pl,
  find_batch_size,
  has_length,
)

class CustomTrainer(Trainer):
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

    # def evaluation_loop(
    #     self,
    #     dataloader: DataLoader,
    #     description: str,
    #     prediction_loss_only: Optional[bool] = None,
    #     ignore_keys: Optional[List[str]] = None,
    #     metric_key_prefix: str = "eval",
    # ) -> EvalLoopOutput:
    #     """
    #     Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

    #     Works both with or without labels.
    #     """
    #     args = self.args

    #     prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

    #     # if eval is called w/o train init deepspeed here
    #     if args.deepspeed and not self.deepspeed:

    #         # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
    #         # from the checkpoint eventually
    #         deepspeed_engine, _, _ = deepspeed_init(
    #             self, num_training_steps=0, resume_from_checkpoint=None, inference=True
    #         )
    #         self.model = deepspeed_engine.module
    #         self.model_wrapped = deepspeed_engine
    #         self.deepspeed = deepspeed_engine

    #     model = self._wrap_model(self.model, training=False, dataloader=dataloader)

    #     # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
    #     # while ``train`` is running, cast it to the right dtype first and then put on device
    #     if not self.is_in_train:
    #         if args.fp16_full_eval:
    #             model = model.to(dtype=torch.float16, device=args.device)
    #         elif args.bf16_full_eval:
    #             model = model.to(dtype=torch.bfloat16, device=args.device)

    #     batch_size = self.args.eval_batch_size

    #     logger.info(f"***** Running {description} *****")
    #     if has_length(dataloader):
    #         logger.info(f"  Num examples = {self.num_examples(dataloader)}")
    #     else:
    #         logger.info("  Num examples: Unknown")
    #     logger.info(f"  Batch size = {batch_size}")

    #     model.eval()

    #     self.callback_handler.eval_dataloader = dataloader
    #     # Do this before wrapping.
    #     eval_dataset = getattr(dataloader, "dataset", None)

    #     if is_torch_tpu_available():
    #         dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

    #     if args.past_index >= 0:
    #         self._past = None

    #     # Initialize containers
    #     # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
    #     losses_host = None
    #     preds_host = None
    #     labels_host = None
    #     inputs_host = None

    #     # losses/preds/labels on CPU (final containers)
    #     all_losses = None
    #     all_preds = None
    #     all_labels = None
    #     all_inputs = None
    #     # Will be useful when we have an iterable dataset so don't know its length.

    #     observed_num_examples = 0
    #     # Main evaluation loop
    #     for step, inputs in enumerate(dataloader):
    #         # Update the observed num examples
    #         observed_batch_size = find_batch_size(inputs)
    #         if observed_batch_size is not None:
    #             observed_num_examples += observed_batch_size
    #             # For batch samplers, batch_size is not known by the dataloader in advance.
    #             if batch_size is None:
    #                 batch_size = observed_batch_size

    #         # Prediction step
    #         loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
    #         inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

    #         if is_torch_tpu_available():
    #             xm.mark_step()

    #         # Update containers on host
    #         if loss is not None:
    #             losses = self._nested_gather(loss.repeat(batch_size))
    #             losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
    #         if labels is not None:
    #             labels = self._pad_across_processes(labels)
    #             labels = self._nested_gather(labels)
    #             labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
    #         if inputs_decode is not None:
    #             inputs_decode = self._pad_across_processes(inputs_decode)
    #             inputs_decode = self._nested_gather(inputs_decode)
    #             inputs_host = (
    #                 inputs_decode
    #                 if inputs_host is None
    #                 else nested_concat(inputs_host, inputs_decode, padding_index=-100)
    #             )
    #         if logits is not None:
    #             logits = self._pad_across_processes(logits)
    #             logits = self._nested_gather(logits)
    #             if self.preprocess_logits_for_metrics is not None:
    #                 logits = self.preprocess_logits_for_metrics(logits, labels)
    #             preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
    #         self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

    #         # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
    #         if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
    #             if losses_host is not None:
    #                 losses = nested_numpify(losses_host)
    #                 all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #             if preds_host is not None:
    #                 logits = nested_numpify(preds_host)
    #                 all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #             if inputs_host is not None:
    #                 inputs_decode = nested_numpify(inputs_host)
    #                 all_inputs = (
    #                     inputs_decode
    #                     if all_inputs is None
    #                     else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #                 )
    #             if labels_host is not None:
    #                 labels = nested_numpify(labels_host)
    #                 all_labels = (
    #                     labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
    #                 )

    #             # Set back to None to begin a new accumulation
    #             losses_host, preds_host, inputs_host, labels_host = None, None, None, None

    #     if args.past_index and hasattr(self, "_past"):
    #         # Clean the state at the end of the evaluation loop
    #         delattr(self, "_past")

    #     # Gather all remaining tensors and put them back on the CPU
    #     if losses_host is not None:
    #         losses = nested_numpify(losses_host)
    #         all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
    #     if preds_host is not None:
    #         logits = nested_numpify(preds_host)
    #         all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
    #     if inputs_host is not None:
    #         inputs_decode = nested_numpify(inputs_host)
    #         all_inputs = (
    #             inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
    #         )
    #     if labels_host is not None:
    #         labels = nested_numpify(labels_host)
    #         all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

    #     # Number of samples
    #     if has_length(eval_dataset):
    #         num_samples = len(eval_dataset)
    #     # The instance check is weird and does not actually check for the type, but whether the dataset has the right
    #     # methods. Therefore we need to make sure it also has the attribute.
    #     elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
    #         num_samples = eval_dataset.num_examples
    #     else:
    #         if has_length(dataloader):
    #             num_samples = self.num_examples(dataloader)
    #         else:  # both len(dataloader.dataset) and len(dataloader) fail
    #             num_samples = observed_num_examples
    #     if num_samples == 0 and observed_num_examples > 0:
    #         num_samples = observed_num_examples

    #     # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
    #     # samplers has been rounded to a multiple of batch_size, so we truncate.
    #     if all_losses is not None:
    #         all_losses = all_losses[:num_samples]
    #     if all_preds is not None:
    #         all_preds = nested_truncate(all_preds, num_samples)
    #     if all_labels is not None:
    #         all_labels = nested_truncate(all_labels, num_samples)
    #     if all_inputs is not None:
    #         all_inputs = nested_truncate(all_inputs, num_samples)

    #     # Metrics!
    #     if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
    #         if args.include_inputs_for_metrics:
    #             metrics = self.compute_metrics(
    #                 EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
    #             )
    #         else:
    #             metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
    #     else:
    #         metrics = {}

    #     # To be JSON-serializable, we need to remove numpy types or zero-d tensors
    #     metrics = denumpify_detensorize(metrics)

    #     if all_losses is not None:
    #         metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

    #     # Prefix all keys with metric_key_prefix + '_'
    #     for key in list(metrics.keys()):
    #         if not key.startswith(f"{metric_key_prefix}_"):
    #             metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

    #     return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


if "crf_4cls" in eval_config.model_name:
  print("Predicting CRF 4 Class")
  if eval_config.with_ground:
    # for key in tokenized_train_dataset.keys():
    for key in ["validation"]:
      with open(
        # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
      ) as f:
        f.write(f"Model Name: {suffix}, Dataset: {key}\n")

      temp_dataset = tokenized_train_dataset[key]
      temp_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels", "prediction_mask"],
        output_all_columns=True,
        device="cuda",
      )
      predictions = []
      logits_all = []

      input_ids = temp_dataset["input_ids"]
      attention_mask = temp_dataset["attention_mask"]
      prediction_mask = temp_dataset["prediction_mask"]
      for i in tqdm(range(len(input_ids))):
        # print(prediction_mask[i])
        logits, predicts = model(
          input_ids=input_ids[i].reshape(1, -1),
          attention_mask=attention_mask[i].reshape(1, -1),
          prediction_mask=prediction_mask[i].reshape(1, -1),
        )
        predictions += predicts
        logits_all += logits.detach()
      
      offset_mapping = temp_dataset["offset_mapping"]
      predicted_spans = []
      predicted_spans_I = []
      predicted_spans_E = []
      # import pdb; pdb.set_trace()
      preds_s = torch.softmax(torch.stack(logits_all), axis=2)

      for i, (preds, probas) in enumerate(zip(predictions, preds_s)):
        predicted_spans.append([])
        predicted_spans_I.append([])
        predicted_spans_E.append([])
        k = 0
        for j, offsets in enumerate(offset_mapping[i]):
          if prediction_mask[i][j] == 0:
            break
          else:
            if k >= len(preds):
              break
            
            # import pdb; pdb.set_trace()
            threshold = 0.8
            if probas[k][preds[k]] < threshold:
              continue

            if preds[k] == 1:
              predicted_spans[-1] += list(range(offsets[0], offsets[1]))
            elif preds[k] == 2:
              predicted_spans_I[-1] += list(range(offsets[0], offsets[1]))
            elif preds[k] == 3:
              predicted_spans_E[-1] += list(range(offsets[0], offsets[1]))
            
            k += 1

      spans = [eval(temp_dataset[i]["spans"]) for i in range(len(temp_dataset))]

      avg_f1_score = np.mean(
        [f1(preds, ground) for preds, ground in zip(predicted_spans, spans)]
      )
      edit_scores = []

      with open(
        os.path.join(eval_config.save_dir, f"spans-pred-{key}_{suffix}.txt"), "w"
      ) as f:
        for i, (pred, pred_I, pred_E) in tqdm(
          enumerate(zip(predicted_spans, predicted_spans_I, predicted_spans_E)), total=len(predicted_spans)):
          # if i == len(preds) - 1:
          #   f.write(f"{i}\t{str(pred)}")
          # else:
          if 1:
            f.write(f"{i}\t{str(pred)}\t{str(pred_I)}\t{str(pred_E)}\n")

           # Edit distance
          ranges = get_ranges(pred)
          ranges_I = get_ranges(pred_I)
          ranges_E = get_ranges(pred_E)
          # print("ranges: ", ranges)
          # Get substring from span range
          range_merged = []
          for i, rng_I in enumerate(ranges_I):
            if rng_I is None:
              continue
            for j, rng_B in enumerate(ranges):
              if rng_B is None:
                continue

              # print(rng_B, rng_I)
              if (rng_I.start == (rng_B.stop + 2)) and (rng_B.stop + 1 not in predicted_spans): 
                # handle one space between them
                new_rng = range(rng_B.start, rng_I.stop)
                # print(new_rng)
                range_merged.append(new_rng)
                ranges_I[i] = None
                ranges[j] = None
                # exit()
                break
          
          # Filter all Nones from ranges and ranges_I
          ranges = [rng for rng in ranges if rng is not None]
          ranges_I = [rng for rng in ranges_I if rng is not None]

          ranges = ranges + ranges_I + range_merged + ranges_E
          # print("ranges: ", ranges)
          # Sort ranges by start
          ranges.sort(key=lambda x: x.start)
            
          sentence = untokenized_train_dataset[key]["sentence"][i]
          gt       = untokenized_train_dataset[key]["gt"][i]
          output = ""
          prev_s = 0
          for i, span in enumerate(ranges):
            # if type(span) == int:
            #   s = span
            #   e = span + 1
            # else:
            if 1:
              s = span.start
              e = span.stop + 1
            
            if s >= len(sentence):
              break

            if s in predicted_spans_E:
              output += sentence[prev_s:e] + "$$"
            else:
              output += sentence[prev_s:s] + "$" + sentence[s:e] + "$"
            prev_s = e
            
          if prev_s < len(sentence):
            output += sentence[prev_s:]
          
          output = replace_fixed(output)
          output = replace_end(output)
          
          # print("output: ", output)
          # gt = untokenized_train_dataset[key]["gt"][i]
          # save output and gt to file
          # with open(os.path.join(eval_config.save_dir, f"output_{key}_{suffix}.txt"), "a+") as outfile:
          #     outfile.write(gt + "\n")
          #     # outfile.write(sentence + "\n")
          #     outfile.write(output + "\n")
          #     outfile.write("\n")

          edit = edit_distance(output, gt)
          edit_scores.append(edit)

      with open(
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
      ) as f:
        f.write(str(avg_f1_score))
        f.write(" ")
        f.write(str(np.mean(edit_scores)))
        f.write("\n")
  else:
    for key in tokenized_test_dataset.keys():
      temp_dataset = tokenized_test_dataset[key]
      temp_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels", "prediction_mask"],
        output_all_columns=True,
        device="cuda",
      )
      predictions = []
      logits_all = []

      input_ids = temp_dataset["input_ids"]
      attention_mask = temp_dataset["attention_mask"]
      prediction_mask = temp_dataset["prediction_mask"]
      for i in tqdm(range(len(input_ids))):
        # print(prediction_mask[i])
        logits, predicts = model(
          input_ids=input_ids[i].reshape(1, -1),
          attention_mask=attention_mask[i].reshape(1, -1),
          prediction_mask=prediction_mask[i].reshape(1, -1),
        )
        predictions += predicts
        logits_all += logits

      offset_mapping = temp_dataset["offset_mapping"]
      predicted_spans = []
      predicted_spans_I = []
      predicted_spans_E = []
      preds_s = torch.softmax(torch.stack(logits_all), axis=2)
      for i, (preds, probas) in enumerate(zip(predictions, preds_s)):
        predicted_spans.append([])
        predicted_spans_I.append([])
        k = 0
        for j, offsets in enumerate(offset_mapping[i]):
          if prediction_mask[i][j] == 0:
            break
          else:
            if k >= len(preds):
              break

            # import pdb; pdb.set_trace()
            threshold = 0.8
            if probas[k][preds[k]] < threshold:
              continue

            if preds[k] == 1:
              predicted_spans[-1] += list(range(offsets[0], offsets[1]))
            elif preds[k] == 2:
              predicted_spans_I[-1] += list(range(offsets[0], offsets[1]))
            elif preds[k] == 3:
              predicted_spans_E[-1] += list(range(offsets[0], offsets[1]))
            k += 1
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred-{key}_{suffix}.txt"), "w"
      ) as f:
        for i, (pred, pred_I, pred_E) in enumerate(
          zip(predicted_spans, predicted_spans_I, predicted_spans_E)):
          # if i == len(preds) - 1:
          #   f.write(f"{i}\t{str(pred)}")
          # else:
          if 1:
            f.write(f"{i}\t{str(pred)}\t{str(pred_I)}\t{str(pred_E)}\n")

elif "crf_3cls" in eval_config.model_name:
  print("Predicting CRF 3 Class")
  if eval_config.with_ground:
    for key in tokenized_train_dataset.keys():
      with open(
        # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
      ) as f:
        f.write(f"Model Name: {suffix}, Dataset: {key}\n")

      temp_dataset = tokenized_train_dataset[key]
      temp_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels", "prediction_mask"],
        output_all_columns=True,
        device="cuda",
      )
      predictions = []

      input_ids = temp_dataset["input_ids"]
      attention_mask = temp_dataset["attention_mask"]
      prediction_mask = temp_dataset["prediction_mask"]
      for i in tqdm(range(len(input_ids))):
        # print(prediction_mask[i])
        predicts = model(
          input_ids=input_ids[i].reshape(1, -1),
          attention_mask=attention_mask[i].reshape(1, -1),
          prediction_mask=prediction_mask[i].reshape(1, -1),
        )[1]
        predictions += predicts
      offset_mapping = temp_dataset["offset_mapping"]
      predicted_spans = []
      predicted_spans_I = []
      for i, preds in enumerate(predictions):
        predicted_spans.append([])
        predicted_spans_I.append([])
        k = 0
        for j, offsets in enumerate(offset_mapping[i]):
          if prediction_mask[i][j] == 0:
            break
          else:
            if k >= len(preds):
              break
            if preds[k] == 1:
              predicted_spans[-1] += list(range(offsets[0], offsets[1]))
            elif preds[k] == 2:
              predicted_spans_I[-1] += list(range(offsets[0], offsets[1]))
            
            k += 1

      spans = [eval(temp_dataset[i]["spans"]) for i in range(len(temp_dataset))]

      avg_f1_score = np.mean(
        [f1(preds, ground) for preds, ground in zip(predicted_spans, spans)]
      )
      edit_scores = []

      with open(
        os.path.join(eval_config.save_dir, f"spans-pred-{key}_{suffix}.txt"), "w"
      ) as f:
        for i, (pred, pred_I) in tqdm(enumerate(zip(predicted_spans, predicted_spans_I)), total=len(predicted_spans)):
          # if i == len(preds) - 1:
          #   f.write(f"{i}\t{str(pred)}")
          # else:
          if 1:
            f.write(f"{i}\t{str(pred)}\t{str(pred_I)}\n")

           # Edit distance
          ranges = get_ranges(pred)
          ranges_I = get_ranges(pred_I)
          # Get substring from span range
          range_merged = []
          for i, rng_I in enumerate(ranges_I):
            if rng_I is None:
              continue
            for j, rng_B in enumerate(ranges):
              if rng_B is None:
                continue

              # print(rng_B, rng_I)
              if (rng_I.start == (rng_B.stop + 2)) and (rng_B.stop + 1 not in pred): 
                # handle one space between them
                new_rng = range(rng_B.start, rng_I.stop)
                # print(new_rng)
                range_merged.append(new_rng)
                ranges_I[i] = None
                ranges[j] = None
                # exit()
                break
          
          # Filter all Nones from ranges and ranges_I
          ranges = [rng for rng in ranges if rng is not None]
          ranges_I = [rng for rng in ranges_I if rng is not None]

          ranges = ranges + ranges_I + range_merged
          # print("ranges: ", ranges)
          # Sort ranges by start
          ranges.sort(key=lambda x: x.start)
          # print("ranges: ", ranges)
          # Get substring from span range
          sentence = untokenized_train_dataset[key]["sentence"][i]
          gt       = untokenized_train_dataset[key]["gt"][i]
          output = ""
          prev_s = 0
          for i, span in enumerate(ranges):
            if type(span) == int:
              s = span
              e = span + 1
            else:
              s = span.start
              e = span.stop + 1
            if s >= len(sentence):
              break
            
            output += sentence[prev_s:s] + "$" + sentence[s:e] + "$"
            prev_s = e
            
          if prev_s < len(sentence):
            output += sentence[prev_s:]
          
          output = replace_fixed(output)
          output = replace_end(output)
          
          # print("output: ", output)
          # gt = untokenized_train_dataset[key]["gt"][i]
          # save output and gt to file
          # with open(os.path.join(eval_config.save_dir, f"output_{key}_{suffix}.txt"), "a+") as outfile:
          #     outfile.write(gt + "\n")
          #     # outfile.write(sentence + "\n")
          #     outfile.write(output + "\n")
          #     outfile.write("\n")

          edit = edit_distance(output, gt)
          edit_scores.append(edit)

      with open(
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
      ) as f:
        f.write(str(avg_f1_score))
        f.write(" ")
        f.write(str(np.mean(edit_scores)))
        f.write("\n")
  else:
    for key in tokenized_test_dataset.keys():
      temp_dataset = tokenized_test_dataset[key]
      temp_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels", "prediction_mask"],
        output_all_columns=True,
        device="cuda",
      )
      predictions = []

      input_ids = temp_dataset["input_ids"]
      attention_mask = temp_dataset["attention_mask"]
      prediction_mask = temp_dataset["prediction_mask"]
      for i in tqdm(range(len(input_ids))):
        # print(prediction_mask[i])
        predicts = model(
          input_ids=input_ids[i].reshape(1, -1),
          attention_mask=attention_mask[i].reshape(1, -1),
          prediction_mask=prediction_mask[i].reshape(1, -1),
        )[1]
        predictions += predicts
      offset_mapping = temp_dataset["offset_mapping"]
      predicted_spans = []
      predicted_spans_I = []
      for i, preds in enumerate(predictions):
        predicted_spans.append([])
        predicted_spans_I.append([])
        k = 0
        for j, offsets in enumerate(offset_mapping[i]):
          if prediction_mask[i][j] == 0:
            break
          else:
            if k >= len(preds):
              break
            if preds[k] == 1:
              predicted_spans[-1] += list(range(offsets[0], offsets[1]))
            elif preds[k] == 2:
              predicted_spans_I[-1] += list(range(offsets[0], offsets[1]))
            k += 1
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred-{key}_{suffix}.txt"), "w"
      ) as f:
        for i, (pred, pred_I) in enumerate(zip(predicted_spans, predicted_spans_I)):
          # if i == len(preds) - 1:
          #   f.write(f"{i}\t{str(pred)}")
          # else:
          if 1:
            f.write(f"{i}\t{str(pred)}\t{str(pred_I)}\n")


elif "crf" in eval_config.model_name:
  if eval_config.with_ground:
    for key in tokenized_train_dataset.keys():
      with open(
        # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
      ) as f:
        f.write(f"Model Name: {suffix}, Dataset: {key}\n")

      temp_dataset = tokenized_train_dataset[key]
      temp_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels", "prediction_mask"],
        output_all_columns=True,
        device="cuda",
      )
      predictions = []

      input_ids = temp_dataset["input_ids"]
      attention_mask = temp_dataset["attention_mask"]
      prediction_mask = temp_dataset["prediction_mask"]
      for i in tqdm(range(len(input_ids))):
        # print(prediction_mask[i])
        predicts = model(
          input_ids=input_ids[i].reshape(1, -1),
          attention_mask=attention_mask[i].reshape(1, -1),
          prediction_mask=prediction_mask[i].reshape(1, -1),
        )[1]
        predictions += predicts
      offset_mapping = temp_dataset["offset_mapping"]
      predicted_spans = []
      for i, preds in enumerate(predictions):
        predicted_spans.append([])
        k = 0
        for j, offsets in enumerate(offset_mapping[i]):
          if prediction_mask[i][j] == 0:
            break
          else:
            if k >= len(preds):
              break
            if preds[k] == 1:
              predicted_spans[-1] += list(range(offsets[0], offsets[1]))
            k += 1

      spans = [eval(temp_dataset[i]["spans"]) for i in range(len(temp_dataset))]

      avg_f1_score = np.mean(
        [f1(preds, ground) for preds, ground in zip(predicted_spans, spans)]
      )
      edit_scores = []

      with open(
        os.path.join(eval_config.save_dir, f"spans-pred-{key}_{suffix}.txt"), "w"
      ) as f:
        for i, pred in tqdm(enumerate(predicted_spans), total=len(predicted_spans)):
          if i == len(preds) - 1:
            f.write(f"{i}\t{str(pred)}")
          else:
            f.write(f"{i}\t{str(pred)}\n")

           # Edit distance
          ranges = get_ranges(pred)
          # print("ranges: ", ranges)
          # Get substring from span range
          sentence = untokenized_train_dataset[key]["sentence"][i]
          gt       = untokenized_train_dataset[key]["gt"][i]
          output = ""
          prev_s = 0
          for i, span in enumerate(ranges):
            if type(span) == int:
              s = span
              e = span + 1
            else:
              s = span.start
              e = span.stop + 1
            
            output += sentence[prev_s:s] + "$" + sentence[s:e] + "$"
            prev_s = e
            
          if prev_s < len(sentence):
            output += sentence[prev_s:]
          
          output = replace_fixed(output)
          output = replace_end(output)
          
          # print("output: ", output)
          # gt = untokenized_train_dataset[key]["gt"][i]
          # save output and gt to file
          # with open(os.path.join(eval_config.save_dir, f"output_{key}_{suffix}.txt"), "a+") as outfile:
          #     outfile.write(gt + "\n")
          #     # outfile.write(sentence + "\n")
          #     outfile.write(output + "\n")
          #     outfile.write("\n")

          edit = edit_distance(output, gt)
          edit_scores.append(edit)

      with open(
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
      ) as f:
        f.write(str(avg_f1_score))
        f.write(" ")
        f.write(str(np.mean(edit_scores)))
        f.write("\n")
  else:
    for key in tokenized_test_dataset.keys():
      temp_dataset = tokenized_test_dataset[key]
      temp_dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels", "prediction_mask"],
        output_all_columns=True,
        device="cuda",
      )
      predictions = []

      input_ids = temp_dataset["input_ids"]
      attention_mask = temp_dataset["attention_mask"]
      prediction_mask = temp_dataset["prediction_mask"]
      for i in tqdm(range(len(input_ids))):
        # print(prediction_mask[i])
        predicts = model(
          input_ids=input_ids[i].reshape(1, -1),
          attention_mask=attention_mask[i].reshape(1, -1),
          prediction_mask=prediction_mask[i].reshape(1, -1),
        )[1]
        predictions += predicts
      offset_mapping = temp_dataset["offset_mapping"]
      predicted_spans = []
      for i, preds in enumerate(predictions):
        predicted_spans.append([])
        k = 0
        for j, offsets in enumerate(offset_mapping[i]):
          if prediction_mask[i][j] == 0:
            break
          else:
            if k >= len(preds):
              break
            if preds[k] == 1:
              predicted_spans[-1] += list(range(offsets[0], offsets[1]))
            k += 1
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred-{key}_{suffix}.txt"), "w"
      ) as f:
        for i, pred in enumerate(predicted_spans):
          if i == len(preds) - 1:
            f.write(f"{i}\t{str(pred)}")
          else:
            f.write(f"{i}\t{str(pred)}\n")


elif "multi" in eval_config.model_name:
  if os.path.exists(os.path.join(eval_config.save_dir, f"thresh.txt")):
    with open(os.path.join(eval_config.save_dir, f"thresh.txt")) as f:
      best_threshold = float(f.read().split("\n")[0])
  else:
    intermediate_eval = untokenized_train_dataset["validation"].map(
      dataset.create_test_features,
      batched=True,
      batch_size=len(untokenized_train_dataset["validation"]),
      remove_columns=untokenized_train_dataset["validation"].column_names,
    )
    tokenized_eval = intermediate_eval.map(
      dataset.prepare_test_features,
      batched=True,
      remove_columns=intermediate_eval.column_names,
    )

    validation_predictions = predict_multi_spans(
      model, tokenized_eval, intermediate_eval, tokenizer
    )

    val_original = untokenized_train_dataset["validation"]
    best_threshold = -1
    best_macro_f1 = -1
    thresholds = np.linspace(0, 1, 100)
    for threshold in tqdm(thresholds):
      macro_f1 = 0
      for row_number in range(len(val_original)):
        row = val_original[row_number]
        ground_spans = eval(row["spans"])
        predicted_spans = validation_predictions[str(row_number)]
        predicted_spans = [
          span
          for span in predicted_spans
          if torch.sigmoid(torch.tensor(span["score"])) > threshold
        ]

        final_predicted_spans = []
        for span in predicted_spans:
          # print(span['start'])
          if span["start"] is not None and span["end"] is not None:
            final_predicted_spans += list(range(span["start"], span["end"]))

        final_predicted_spans = sorted(final_predicted_spans)
        macro_f1 += f1(final_predicted_spans, ground_spans)
      avg = macro_f1 / len(val_original)
      if avg > best_macro_f1:
        best_macro_f1 = avg
        best_threshold = threshold
    with open(os.path.join(eval_config.save_dir, f"thresh.txt"), "w") as f:
      f.write(str(best_threshold) + "\n")
      f.write(str(best_macro_f1))

  topk = eval_config.topk

  if eval_config.with_ground:
    for key in untokenized_train_dataset.keys():
      f1_scores = []
      intermediate_test = untokenized_train_dataset[key].map(
        dataset.create_test_features,
        batched=True,
        batch_size=len(untokenized_train_dataset[key]),
        remove_columns=untokenized_train_dataset[key].column_names,
      )
      tokenized_test = intermediate_test.map(
        dataset.prepare_test_features,
        batched=True,
        remove_columns=intermediate_test.column_names,
      )

      test_predictions = predict_multi_spans(
        model, tokenized_test, intermediate_test, tokenizer
      )

      test_original = untokenized_train_dataset[key]
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
      ) as f:
        for row_number in range(len(test_original)):
          row = test_original[row_number]
          ground_spans = eval(row["spans"])
          predicted_spans = test_predictions[str(row_number)]
          predicted_spans = [
            span
            for span in predicted_spans
            if torch.sigmoid(torch.tensor(span["score"])) > best_threshold
          ]

          final_predicted_spans = []
          for span in predicted_spans:
            # print(span['start'])
            if span["start"] is not None and span["end"] is not None:
              final_predicted_spans += list(
                range(span["start"], span["end"])
              )

          final_predicted_spans = sorted(final_predicted_spans)
          if row_number != len(test_original) - 1:
            f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
          else:
            f.write(f"{row_number}\t{str(final_predicted_spans)}")
          f1_scores.append(f1(final_predicted_spans, eval(row["spans"])))
      with open(
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "w"
      ) as f:
        f.write(str(np.mean(f1_scores)))

  else:
    for key in untokenized_test_dataset.keys():
      intermediate_test = untokenized_test_dataset[key].map(
        dataset.create_test_features,
        batched=True,
        batch_size=len(untokenized_test_dataset[key]),
        remove_columns=untokenized_test_dataset[key].column_names,
      )
      tokenized_test = intermediate_test.map(
        dataset.prepare_test_features,
        batched=True,
        remove_columns=intermediate_test.column_names,
      )

      test_predictions = predict_tokens_spans(
        model, tokenized_test, intermediate_test, tokenizer
      )

      test_original = untokenized_test_dataset[key]
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
      ) as f:
        for row_number in range(len(test_original)):
          row = test_original[row_number]
          ground_spans = eval(row["spans"])
          predicted_spans = test_predictions[str(row_number)]
          predicted_spans = [
            span
            for span in predicted_spans
            if torch.sigmoid(torch.tensor(span["score"])) > best_threshold
          ]

          final_predicted_spans = []
          for span in predicted_spans:
            # print(span['start'])
            if span["start"] is not None and span["end"] is not None:
              final_predicted_spans += list(
                range(span["start"], span["end"])
              )

          final_predicted_spans = sorted(final_predicted_spans)
          if row_number != len(test_original) - 1:
            f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
          else:
            f.write(f"{row_number}\t{str(final_predicted_spans)}")

elif "token_spans" in eval_config.model_name:

  if eval_config.style is None:
    # print("No style specified, using default")

    if os.path.exists(os.path.join(eval_config.save_dir, f"thresh.txt")):
      with open(os.path.join(eval_config.save_dir, f"thresh.txt")) as f:
        best_threshold = float(f.read().split("\n")[0])
    else:
      intermediate_eval = untokenized_train_dataset["validation"].map(
        dataset.create_test_features,
        batched=True,
        batch_size=len(untokenized_train_dataset["validation"]),
        remove_columns=untokenized_train_dataset["validation"].column_names,
      )
      tokenized_eval = intermediate_eval.map(
        dataset.prepare_test_features,
        batched=True,
        remove_columns=intermediate_eval.column_names,
      )

      validation_predictions = predict_tokens_spans(
        model, tokenized_eval, intermediate_eval, tokenizer
      )

      val_original = untokenized_train_dataset["validation"]
      best_threshold = -1
      best_macro_f1 = -1
      thresholds = np.linspace(0, 1, 100)
      for threshold in tqdm(thresholds):
        macro_f1 = 0
        for row_number in range(len(val_original)):
          row = val_original[row_number]
          ground_spans = eval(row["spans"])
          predicted_spans = validation_predictions[str(row_number)]
          predicted_spans = [
            span
            for span in predicted_spans
            if torch.sigmoid(torch.tensor(span["score"])) > threshold
          ]

          final_predicted_spans = []
          for span in predicted_spans:
            # print(span['start'])
            if span["start"] is not None and span["end"] is not None:
              final_predicted_spans += list(
                range(span["start"], span["end"])
              )

          final_predicted_spans = sorted(final_predicted_spans)
          macro_f1 += f1(final_predicted_spans, ground_spans)
        avg = macro_f1 / len(val_original)
        if avg > best_macro_f1:
          best_macro_f1 = avg
          best_threshold = threshold
      with open(os.path.join(eval_config.save_dir, f"thresh.txt"), "w") as f:
        f.write(str(best_threshold) + "\n")
        f.write(str(best_macro_f1))

    topk = eval_config.topk

    if eval_config.with_ground:
      for key in untokenized_train_dataset.keys():
        f1_scores = []
        intermediate_test = untokenized_train_dataset[key].map(
          dataset.create_test_features,
          batched=True,
          batch_size=len(untokenized_train_dataset[key]),
          remove_columns=untokenized_train_dataset[key].column_names,
        )
        tokenized_test = intermediate_test.map(
          dataset.prepare_test_features,
          batched=True,
          remove_columns=intermediate_test.column_names,
        )

        test_predictions = predict_tokens_spans(
          model, tokenized_test, intermediate_test, tokenizer
        )

        test_original = untokenized_train_dataset[key]
        with open(
          os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
        ) as f:
          for row_number in range(len(test_original)):
            row = test_original[row_number]
            ground_spans = eval(row["spans"])
            predicted_spans = test_predictions[str(row_number)]
            predicted_spans = [
              span
              for span in predicted_spans
              if torch.sigmoid(torch.tensor(span["score"]))
              > best_threshold
            ]

            final_predicted_spans = []
            for span in predicted_spans:
              # print(span['start'])
              if span["start"] is not None and span["end"] is not None:
                final_predicted_spans += list(
                  range(span["start"], span["end"])
                )

            final_predicted_spans = sorted(final_predicted_spans)
            if row_number != len(test_original) - 1:
              f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
            else:
              f.write(f"{row_number}\t{str(final_predicted_spans)}")
            f1_scores.append(f1(final_predicted_spans, eval(row["spans"])))
        with open(
          os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "w"
        ) as f:
          f.write(str(np.mean(f1_scores)))

    else:
      for key in untokenized_test_dataset.keys():
        intermediate_test = untokenized_test_dataset[key].map(
          dataset.create_test_features,
          batched=True,
          batch_size=len(untokenized_test_dataset[key]),
          remove_columns=untokenized_test_dataset[key].column_names,
        )
        tokenized_test = intermediate_test.map(
          dataset.prepare_test_features,
          batched=True,
          remove_columns=intermediate_test.column_names,
        )

        test_predictions = predict_tokens_spans(
          model, tokenized_test, intermediate_test, tokenizer
        )

        test_original = untokenized_test_dataset[key]
        with open(
          os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
        ) as f:
          for row_number in range(len(test_original)):
            row = test_original[row_number]
            ground_spans = eval(row["spans"])
            predicted_spans = test_predictions[str(row_number)]
            predicted_spans = [
              span
              for span in predicted_spans
              if torch.sigmoid(torch.tensor(span["score"]))
              > best_threshold
            ]

            final_predicted_spans = []
            for span in predicted_spans:
              # print(span['start'])
              if span["start"] is not None and span["end"] is not None:
                final_predicted_spans += list(
                  range(span["start"], span["end"])
                )

            final_predicted_spans = sorted(final_predicted_spans)
            if row_number != len(test_original) - 1:
              f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
            else:
              f.write(f"{row_number}\t{str(final_predicted_spans)}")

  elif eval_config.style == "token":
    if eval_config.with_ground:
      for key in tokenized_train_dataset.keys():

        untokenized_dataset = untokenized_train_dataset[key]
        temp_untokenized_spans = untokenized_dataset["spans"]
        temp_intermediate_dataset = untokenized_dataset.map(
          dataset.create_test_features,
          batched=True,
          batch_size=1000000,  ##Unusually Large Batch Size ## Needed For Correct ID mapping
          remove_columns=untokenized_dataset.column_names,
        )

        temp_tokenized_dataset = temp_intermediate_dataset.map(
          dataset.prepare_test_features,
          batched=True,
          remove_columns=temp_intermediate_dataset.column_names,
        )
        temp_offset_mapping = temp_tokenized_dataset["offset_mapping"]

        preds = get_token_spans_separate_logits(
          model, temp_tokenized_dataset, type="token"
        )  ## Token Logits
        preds = np.argmax(preds, axis=2)
        f1_scores = []
        with open(
          os.path.join(eval_config.save_dir, f"spans-pred_{key}.txt"), "w"
        ) as f:
          for i, pred in tqdm(enumerate(preds)):
            # print(key,i)
            ## Batch Wise
            # print(len(prediction))
            predicted_spans = []
            for j, tokenwise_prediction in enumerate(
              pred[: len(temp_offset_mapping[i])]
            ):
              if (
                temp_offset_mapping[i][j] is not None
                and tokenwise_prediction == 1
              ):  # question tokens have None offset.
                predicted_spans += list(
                  range(
                    temp_offset_mapping[i][j][0],
                    temp_offset_mapping[i][j][1],
                  )
                )
            if i == len(preds) - 1:
              f.write(f"{i}\t{str(predicted_spans)}")
            else:
              f.write(f"{i}\t{str(predicted_spans)}\n")
            f1_scores.append(
              f1(
                predicted_spans,
                eval(temp_untokenized_spans[i]),
              )
            )
        with open(
          os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "w"
        ) as f:
          f.write(str(np.mean(f1_scores)))
    else:
      for key in tokenized_test_dataset.keys():
        untokenized_dataset = untokenized_test_dataset[key]
        temp_untokenized_spans = untokenized_dataset["spans"]
        temp_intermediate_dataset = untokenized_dataset.map(
          dataset.create_test_features,
          batched=True,
          batch_size=1000000,  ##Unusually Large Batch Size ## Needed For Correct ID mapping
          remove_columns=untokenized_dataset.column_names,
        )

        temp_tokenized_dataset = temp_intermediate_dataset.map(
          dataset.prepare_test_features,
          batched=True,
          remove_columns=temp_intermediate_dataset.column_names,
        )
        temp_offset_mapping = temp_tokenized_dataset["offset_mapping"]

        preds = get_token_spans_separate_logits(
          model, temp_tokenized_dataset, type="token"
        )  ## Token Logits
        preds = np.argmax(preds, axis=2)
        f1_scores = []
        with open(
          os.path.join(eval_config.save_dir, f"spans-pred_{key}.txt"), "w"
        ) as f:
          for i, pred in tqdm(enumerate(preds)):
            # print(key,i)
            ## Batch Wise
            # print(len(prediction))
            predicted_spans = []
            for j, tokenwise_prediction in enumerate(
              pred[: len(temp_offset_mapping[i])]
            ):
              if (
                temp_offset_mapping[i][j] is not None
                and tokenwise_prediction == 1
              ):  # question tokens have None offset.
                predicted_spans += list(
                  range(
                    temp_offset_mapping[i][j][0],
                    temp_offset_mapping[i][j][1],
                  )
                )
            if i == len(preds) - 1:
              f.write(f"{i}\t{str(predicted_spans)}")
            else:
              f.write(f"{i}\t{str(predicted_spans)}\n")

  elif eval_config.style == "spans":
    if os.path.exists(os.path.join(eval_config.save_dir, f"thresh.txt")):
      with open(os.path.join(eval_config.save_dir, f"thresh.txt")) as f:
        best_threshold = float(f.read().split("\n")[0])
    else:
      intermediate_eval = untokenized_train_dataset["validation"].map(
        dataset.create_test_features,
        batched=True,
        batch_size=len(untokenized_train_dataset["validation"]),
        remove_columns=untokenized_train_dataset["validation"].column_names,
      )
      tokenized_eval = intermediate_eval.map(
        dataset.prepare_test_features,
        batched=True,
        remove_columns=intermediate_eval.column_names,
      )
      preds = get_token_spans_separate_logits(model, tokenized_eval, type="spans")
      validation_predictions = postprocess_multi_span_predictions(
        tokenized_eval, intermediate_eval, preds, tokenizer
      )

      val_original = untokenized_train_dataset["validation"]
      best_threshold = -1
      best_macro_f1 = -1
      thresholds = np.linspace(0, 1, 100)
      for threshold in tqdm(thresholds):
        macro_f1 = 0
        for row_number in range(len(val_original)):
          row = val_original[row_number]
          ground_spans = eval(row["spans"])
          predicted_spans = validation_predictions[str(row_number)]
          predicted_spans = [
            span
            for span in predicted_spans
            if torch.sigmoid(torch.tensor(span["score"])) > threshold
          ]

          final_predicted_spans = []
          for span in predicted_spans:
            # print(span['start'])
            if span["start"] is not None and span["end"] is not None:
              final_predicted_spans += list(
                range(span["start"], span["end"])
              )

          final_predicted_spans = sorted(final_predicted_spans)
          macro_f1 += f1(final_predicted_spans, ground_spans)
        avg = macro_f1 / len(val_original)
        if avg > best_macro_f1:
          best_macro_f1 = avg
          best_threshold = threshold
      with open(os.path.join(eval_config.save_dir, f"thresh.txt"), "w") as f:
        f.write(str(best_threshold) + "\n")
        f.write(str(best_macro_f1))

    # topk = eval_config.topk

    if eval_config.with_ground:
      for key in untokenized_train_dataset.keys():
        f1_scores = []
        intermediate_test = untokenized_train_dataset[key].map(
          dataset.create_test_features,
          batched=True,
          batch_size=len(untokenized_train_dataset[key]),
          remove_columns=untokenized_train_dataset[key].column_names,
        )
        tokenized_test = intermediate_test.map(
          dataset.prepare_test_features,
          batched=True,
          remove_columns=intermediate_test.column_names,
        )

        preds = get_token_spans_separate_logits(
          model, tokenized_test, type="spans"
        )
        test_predictions = postprocess_multi_span_predictions(
          tokenized_test, intermediate_test, preds, tokenizer
        )

        test_original = untokenized_train_dataset[key]
        with open(
          os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
        ) as f:
          for row_number in range(len(test_original)):
            row = test_original[row_number]
            ground_spans = eval(row["spans"])
            predicted_spans = test_predictions[str(row_number)]
            predicted_spans = [
              span
              for span in predicted_spans
              if torch.sigmoid(torch.tensor(span["score"]))
              > best_threshold
            ]

            final_predicted_spans = []
            for span in predicted_spans:
              # print(span['start'])
              if span["start"] is not None and span["end"] is not None:
                final_predicted_spans += list(
                  range(span["start"], span["end"])
                )

            final_predicted_spans = sorted(final_predicted_spans)
            if row_number != len(test_original) - 1:
              f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
            else:
              f.write(f"{row_number}\t{str(final_predicted_spans)}")
            f1_scores.append(f1(final_predicted_spans, eval(row["spans"])))
        with open(
          os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "w"
        ) as f:
          f.write(str(np.mean(f1_scores)))

    else:
      for key in untokenized_test_dataset.keys():
        intermediate_test = untokenized_test_dataset[key].map(
          dataset.create_test_features,
          batched=True,
          batch_size=len(untokenized_test_dataset[key]),
          remove_columns=untokenized_test_dataset[key].column_names,
        )
        tokenized_test = intermediate_test.map(
          dataset.prepare_test_features,
          batched=True,
          remove_columns=intermediate_test.column_names,
        )

        preds = get_token_spans_separate_logits(
          model, tokenized_test, type="spans"
        )
        test_predictions = postprocess_multi_span_predictions(
          tokenized_test, intermediate_test, preds, tokenizer
        )

        test_original = untokenized_test_dataset[key]
        with open(
          os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
        ) as f:
          for row_number in range(len(test_original)):
            row = test_original[row_number]
            ground_spans = eval(row["spans"])
            predicted_spans = test_predictions[str(row_number)]
            predicted_spans = [
              span
              for span in predicted_spans
              if torch.sigmoid(torch.tensor(span["score"]))
              > best_threshold
            ]

            final_predicted_spans = []
            for span in predicted_spans:
              # print(span['start'])
              if span["start"] is not None and span["end"] is not None:
                final_predicted_spans += list(
                  range(span["start"], span["end"])
                )

            final_predicted_spans = sorted(final_predicted_spans)
            if row_number != len(test_original) - 1:
              f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
            else:
              f.write(f"{row_number}\t{str(final_predicted_spans)}")


elif "token_3cls" in eval_config.model_name:
  print("Token 3cls eval running...") 
  args = TrainingArguments(output_dir="./out", **eval_config.args)
  trainer = Trainer(
    model=model,
    args=args,
  )
  if eval_config.with_ground:
    # for key in tokenized_train_dataset.keys():
    for key in ['validation']:
      temp_offset_mapping = tokenized_train_dataset[key]["offset_mapping"]
      predictions = trainer.predict(tokenized_train_dataset[key])
      temp_untokenized_spans = untokenized_train_dataset[key]["spans"]
      # print(untokenized_train_dataset[key])

      preds = predictions.predictions
      preds_s = torch.softmax(torch.from_numpy(preds).float(), dim=2)
      preds_proba = np.max(preds_s.numpy(), axis=2)
      preds = np.argmax(preds, axis=2)
      f1_scores = []
      edit_scores = []
      with open(
        # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
      ) as f:
        f.write(f"Model Name: {suffix}, Dataset: {key}\n")

      # outputs = []
      # gts = [] 
      
      with open(
        # os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "w"
        os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "w"
      ) as f:
        for i, (pred, pred_proba) in tqdm(enumerate(zip(preds, preds_proba))):
          # print(key,i)
          ## Batch Wise
          # print(len(prediction))
          predicted_spans = []
          predicted_spans_I = []
          for j, (tokenwise_prediction, token_p) in enumerate(
            zip(
            pred[: len(temp_offset_mapping[i])],
            pred_proba[: len(temp_offset_mapping[i])]
            )
          ):
            thresh = 0.8
            if token_p < thresh: 
              continue
            if tokenwise_prediction == 1:
              predicted_spans += list(
                range(
                  temp_offset_mapping[i][j][0],
                  temp_offset_mapping[i][j][1],
                )
              )
            elif tokenwise_prediction == 2:
              predicted_spans_I += list(
                range(
                  temp_offset_mapping[i][j][0],
                  temp_offset_mapping[i][j][1],
                )
              )
          # if i == len(preds) - 1:
          #     f.write(f"{i}\t{str(predicted_spans)}\t{str(predicted_spans)}")
          # else:
          if 1:
            f.write(f"{i}\t{str(predicted_spans)}\t{str(predicted_spans_I)}\n")
          
          # Edit distance
          ranges = get_ranges(predicted_spans)
          ranges_I = get_ranges(predicted_spans_I)
          # print("ranges: ", ranges)
          # Get substring from span range
          range_merged = []
          for i, rng_I in enumerate(ranges_I):
            if rng_I is None:
              continue
            for j, rng_B in enumerate(ranges):
              if rng_B is None:
                continue

              # print(rng_B, rng_I)
              if (rng_I.start == (rng_B.stop + 2)) and (rng_B.stop + 1 not in predicted_spans): 
                # handle one space between them
                new_rng = range(rng_B.start, rng_I.stop)
                # print(new_rng)
                range_merged.append(new_rng)
                ranges_I[i] = None
                ranges[j] = None
                # exit()
                break
          
          # Filter all Nones from ranges and ranges_I
          ranges = [rng for rng in ranges if rng is not None]
          ranges_I = [rng for rng in ranges_I if rng is not None]

          ranges = ranges + ranges_I + range_merged
          # print("ranges: ", ranges)
          # Sort ranges by start
          ranges.sort(key=lambda x: x.start)
            
          sentence = untokenized_train_dataset[key]["sentence"][i]
          gt       = untokenized_train_dataset[key]["gt"][i]
          output = ""
          prev_s = 0
          for i, span in enumerate(ranges):
            if type(span) == int:
              s = span
              e = span + 1
            else:
              s = span.start
              e = span.stop + 1
            
            if s >= len(sentence):
              break
            
            output += sentence[prev_s:s] + "$" + sentence[s:e] + "$"
            prev_s = e
            
          if prev_s < len(sentence):
            output += sentence[prev_s:]
          
          output = replace_fixed(output)
          output = replace_end(output)
          
          # print("output: ", output)
          # gt = untokenized_train_dataset[key]["gt"][i]
          # save output and gt to file
          # with open(os.path.join(eval_config.save_dir, f"output_{key}_{suffix}.txt"), "a+") as outfile:
          #     outfile.write(gt + "\n")
          #     # outfile.write(sentence + "\n")
          #     outfile.write(output + "\n")
          #     outfile.write("\n")

          edit = edit_distance(output, gt)

          edit_scores.append(edit)
          # outputs.append(output)
          # gts.append(gt)

          f1_scores.append(
            f1(
              predicted_spans,
              eval(temp_untokenized_spans[i]),
            )
          )
        
      # from pandarallel import pandarallel

      # pandarallel.initialize()
      # tqdm.pandas()
      
      # df = pd.DataFrame({"output": outputs, "gt": gts})
      # # Computer edit distance
      # edit_scores = df.parallel_appy(lambda x: edit_distance(x["output"], x["gt"]), axis=1)

      with open(
        # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
      ) as f:
        f.write(str(np.mean(f1_scores)))
        f.write(" ")
        f.write(str(np.mean(edit_scores)))
        f.write("\n")
  else:
    for key in tokenized_test_dataset.keys():
      temp_offset_mapping = tokenized_test_dataset[key]["offset_mapping"]
      predictions = trainer.predict(tokenized_test_dataset[key])
      preds = predictions.predictions
      preds_s = torch.softmax(torch.from_numpy(preds).float(), dim=2)
      preds_proba = np.max(preds_s.numpy(), axis=2)
      preds = np.argmax(preds, axis=2)
      f1_scores = []
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "a"
      ) as f:
        for i, (pred, pred_proba) in tqdm(enumerate(zip(preds, preds_proba))):
          # print(key,i)
          ## Batch Wise
          # print(len(prediction))
          predicted_spans = []
          predicted_spans_I = []
          for j, (tokenwise_prediction, token_p) in enumerate(
            zip(
            pred[: len(temp_offset_mapping[i])],
            pred_proba[: len(temp_offset_mapping[i])]
            )
          ):
            thresh = 0.8
            if token_p < thresh: 
              continue
            if tokenwise_prediction == 1:
              predicted_spans += list(
                range(
                  temp_offset_mapping[i][j][0],
                  temp_offset_mapping[i][j][1],
                )
              )
            elif tokenwise_prediction == 2:
              predicted_spans_I += list(
                range(
                  temp_offset_mapping[i][j][0],
                  temp_offset_mapping[i][j][1],
                )
              )

          # if i == len(preds) - 1:
          #     f.write(f"{i}\t{str(predicted_spans)}")
          # else:
          #     f.write(f"{i}\t{str(predicted_spans)}\n")
          if 1:
            f.write(f"{i}\t{str(predicted_spans)}\t{str(predicted_spans_I)}\n")

elif "token_4cls_v2" in eval_config.model_name:
  print("Token 4cls v2 eval running...") 
  # Set prediction args
  # args.do_predict = True
  args = TrainingArguments(output_dir="./out", **eval_config.args)
  # trainer = Trainer(
  trainer = CustomTrainer(
    model=model,
    args=args,
  )
  if eval_config.with_ground:
    # for key in tokenized_train_dataset.keys():
    for key in ['validation']:
      temp_offset_mapping = tokenized_train_dataset[key]["offset_mapping"]
      predictions = trainer.predict(tokenized_train_dataset[key])
      temp_untokenized_spans = untokenized_train_dataset[key]["spans"]
      # print(untokenized_train_dataset[key])

      preds_all = predictions.predictions      
      # import ipdb; ipdb.set_trace()
      
      # for thresh in np.linspace(0.7, 0.95, 10): # Best is 0.75
      for thresh in [0.8]: 
        f1_scores = []
        edit_scores = []
        with open(
          # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
          os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
        ) as f:
          # f.write(f"Model Name: {suffix}, Dataset: {key}\n")
          f.write(f"Model Name: {suffix}, Dataset: {key}, Threshold: {thresh}\n")

        # outputs = []
        # gts = [] 
        
        with open(
          # os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "w"
          os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "w"
        ) as f:
          preds = preds_all[:, :, :3]
          preds_s = torch.softmax(torch.from_numpy(preds).float(), dim=2)
          preds_proba = np.max(preds_s.numpy(), axis=2)
          preds = np.argmax(preds, axis=2)

          preds_1 = preds_all[:, :, 3:]
          preds_s_1 = torch.softmax(torch.from_numpy(preds_1).float(), dim=2)
          preds_proba_1 = np.max(preds_s_1.numpy(), axis=2)
          preds_1 = np.argmax(preds_1, axis=2)

          for i, (pred, pred_proba, pred_1, pred_proba_1) in tqdm(enumerate(zip(
                                      preds, preds_proba, preds_1, preds_proba_1)), total=len(preds)):
            # print(key,i)
            counterrr = i
            ## Batch Wise
            # print(len(prediction))
            predicted_spans = []
            predicted_spans_I = []
            predicted_spans_E = []
            for j, (tokenwise_prediction, token_p, tokenwise_prediction_1, token_p_1) in enumerate(
              zip(
              pred[: len(temp_offset_mapping[i])],
              pred_proba[: len(temp_offset_mapping[i])],
              pred_1[: len(temp_offset_mapping[i])],
              pred_proba_1[: len(temp_offset_mapping[i])],
              )
            ):
              # thresh = 0.8
              if token_p > thresh: 
                if tokenwise_prediction == 1:
                  predicted_spans += list(
                    range(
                      temp_offset_mapping[i][j][0],
                      temp_offset_mapping[i][j][1],
                    )
                  )
                elif tokenwise_prediction == 2:
                  predicted_spans_I += list(
                    range(
                      temp_offset_mapping[i][j][0],
                      temp_offset_mapping[i][j][1],
                    )
                  )

              if token_p_1 > thresh:
              # if 1:
                if tokenwise_prediction_1 == 1:
                  predicted_spans_E += list(
                    range(
                      temp_offset_mapping[i][j][0],
                      temp_offset_mapping[i][j][1],
                    )
                  )
          
            # if i == len(preds) - 1:
            #     f.write(f"{i}\t{str(predicted_spans)}\t{str(predicted_spans)}")
            # else:
            if 1:
              f.write(f"{i}\t{str(predicted_spans)}\t{str(predicted_spans_I)}\t{str(predicted_spans_E)}\n")
            
            # Edit distance
            ranges = get_ranges(predicted_spans)
            ranges_I = get_ranges(predicted_spans_I)
            ranges_E = get_ranges(predicted_spans_E)
            # print("ranges: ", ranges)
            # Get substring from span range
            range_merged = []
            for i, rng_I in enumerate(ranges_I):
              if rng_I is None:
                continue
              for j, rng_B in enumerate(ranges):
                if rng_B is None:
                  continue

                # print(rng_B, rng_I)
                if (rng_I.start == (rng_B.stop + 2)) and (rng_B.stop + 1 not in predicted_spans): 
                  # handle one space between them
                  new_rng = range(rng_B.start, rng_I.stop)
                  # print(new_rng)
                  range_merged.append(new_rng)
                  ranges_I[i] = None
                  ranges[j] = None
                  # exit()
                  break
            
            # Filter all Nones from ranges and ranges_I
            ranges = [rng for rng in ranges if rng is not None]
            ranges_I = [rng for rng in ranges_I if rng is not None]

            # ranges = ranges + ranges_I + range_merged + ranges_E
            ranges = ranges + ranges_I + range_merged
            # print("ranges: ", ranges)
            # Sort ranges by start
            ranges.sort(key=lambda x: x.start)
              
            sentence = untokenized_train_dataset[key]["sentence"][i]
            gt       = untokenized_train_dataset[key]["gt"][i]
            output = ""
            prev_s = 0
            from collections import defaultdict
            dollar_count = defaultdict(int)
            for i, span in enumerate(ranges):
              # if type(span) == int:
              #   s = span
              #   e = span + 1
              # else:
              if 1:
                s = span.start
                e = span.stop + 1
              
              if s >= len(sentence):
                break

              # if s in predicted_spans_E:
              #   output += sentence[prev_s:e] + "$$"
              # else:
              if 1:
                output += sentence[prev_s:s] + "$" + sentence[s:e] + "$"
                # dollar_count[s:e] += 1
                # Increase dollar count from s to e
                if prev_s - 1 >= 0:
                  dol_p = dollar_count[prev_s - 1]
                else:
                  # if s == 0:
                  #   dol_p = 1
                  # else:
                  if 1:
                    dol_p = 0

                for j in range(prev_s, s):
                  dollar_count[j] = 1 + dol_p
                for j in range(s, e):
                  dollar_count[j] = 2 + dol_p

              prev_s = e
              
            if prev_s < len(sentence):
              if prev_s - 1 >= 0:
                for j in range(prev_s, len(sentence)):
                  dollar_count[j] = dollar_count[prev_s - 1] + 1
              else:
                for j in range(prev_s, len(sentence)):
                  dollar_count[j] = 1

              output += sentence[prev_s:]
            
            output_1 = output
            
            if len(ranges_E) > 0:
            # if 0:
              output_2 = ""
              prev_s = 0
              starts_with_dollar = output.startswith("$")
              for i, span in enumerate(ranges_E):
                # if type(span) == int:
                #   s = span
                #   e = span + 1
                # else:
                if 1:
                  s = span.start
                  e = span.stop + 1
                
                if s >= len(sentence):
                  break

                # if s in predicted_spans_E:
                if 1:
                  if starts_with_dollar and prev_s == 0:
                    dollar1 = 0
                  else:
                  # if 1:
                    dollar1 = dollar_count[prev_s] - 1
                  # dollar1 = dollar_count[prev_s]
                  # dollar2 = dollar_count[e] - 1
                  dollar2 = dollar_count[e-1] - 1
                  # dollar2 = dollar_count[e-1]
                  output_2 += output[prev_s+dollar1:e+dollar2] + "$$"
                # else:
                # if 1:
                  # output += sentence[prev_s:s] + "$" + sentence[s:e] + "$"
                prev_s = e
                
              if prev_s < len(sentence):
                dollar1 = dollar_count[prev_s] - 1
                output_2 += output[prev_s+dollar1:]
              
              output = output_2

            
            if len(ranges_E) > 0:
              # import pdb; pdb.set_trace()
              if counterrr == 1012:
                print(counterrr, gt, "|||", output)
                print(output_1)
                print("dollars: ", dollar_count)

            output = replace_fixed(output)
            output = replace_end(output)

            # print("output: ", output)
            # gt = untokenized_train_dataset[key]["gt"][i]
            # save output and gt to file
            # with open(os.path.join(eval_config.save_dir, f"output_{key}_{suffix}.txt"), "a+") as outfile:
            #     outfile.write(gt + "\n")
            #     # outfile.write(sentence + "\n")
            #     outfile.write(output + "\n")
            #     outfile.write("\n")

            edit = edit_distance(output, gt)
            
            
            edit_scores.append(edit)
            # outputs.append(output)
            # gts.append(gt)

            f1_scores.append(
              f1(
                predicted_spans,
                eval(temp_untokenized_spans[i]),
              )
            )
          
        # from pandarallel import pandarallel

        # pandarallel.initialize()
        # tqdm.pandas()
        
        # df = pd.DataFrame({"output": outputs, "gt": gts})
        # # Computer edit distance
        # edit_scores = df.parallel_appy(lambda x: edit_distance(x["output"], x["gt"]), axis=1)

        with open(
          # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
          os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
        ) as f:
          f.write(str(np.mean(f1_scores)))
          f.write(" ")
          f.write(str(np.mean(edit_scores)))
          f.write("\n")
  else:
    for key in tokenized_test_dataset.keys():
      temp_offset_mapping = tokenized_test_dataset[key]["offset_mapping"]
      predictions = trainer.predict(tokenized_test_dataset[key])
      preds = predictions.predictions
      
      # Find the probability of the predicted label
      # import ipdb; ipdb.set_trace()
      preds_s = torch.softmax(torch.from_numpy(preds).float(), dim=2)
      preds_proba = np.max(preds_s.numpy(), axis=2)

      preds = np.argmax(preds, axis=2)
      preds_all = preds
      f1_scores = []
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "a"
      ) as f:
        preds = preds_all[:, :, :3]
        preds_s = torch.softmax(torch.from_numpy(preds).float(), dim=2)
        preds_proba = np.max(preds_s.numpy(), axis=2)
        preds = np.argmax(preds, axis=2)

        preds_1 = preds_all[:, :, 3:]
        preds_s_1 = torch.softmax(torch.from_numpy(preds_1).float(), dim=2)
        preds_proba_1 = np.max(preds_s_1.numpy(), axis=2)
        preds_1 = np.argmax(preds_1, axis=2)

        for i, (pred, pred_proba, pred_1, pred_proba_1) in tqdm(enumerate(zip(
                                    preds, preds_proba, preds_1, preds_proba_1)), total=len(preds)):
          # print(key,i)
          ## Batch Wise
          # print(len(prediction))
          predicted_spans = []
          predicted_spans_I = []
          predicted_spans_E = []
          p_B, p_I, p_E = [], [], []    
          if 1:
            # print(key,i)
            counterrr = i
            ## Batch Wise
            # print(len(prediction))
            predicted_spans = []
            predicted_spans_I = []
            predicted_spans_E = []
            for j, (tokenwise_prediction, token_p, tokenwise_prediction_1, token_p_1) in enumerate(
              zip(
              pred[: len(temp_offset_mapping[i])],
              pred_proba[: len(temp_offset_mapping[i])],
              pred_1[: len(temp_offset_mapping[i])],
              pred_proba_1[: len(temp_offset_mapping[i])],
              )
            ):
              thresh = 0.8
              if token_p > thresh: 
                if tokenwise_prediction == 1:
                  predicted_spans += list(
                    range(
                      temp_offset_mapping[i][j][0],
                      temp_offset_mapping[i][j][1],
                    )
                  )
                elif tokenwise_prediction == 2:
                  predicted_spans_I += list(
                    range(
                      temp_offset_mapping[i][j][0],
                      temp_offset_mapping[i][j][1],
                    )
                  )

              if token_p_1 > thresh:
              # if 1:
                if tokenwise_prediction_1 == 1:
                  predicted_spans_E += list(
                    range(
                      temp_offset_mapping[i][j][0],
                      temp_offset_mapping[i][j][1],
                    )
                  )
          
            # if i == len(preds) - 1:
            #     f.write(f"{i}\t{str(predicted_spans)}\t{str(predicted_spans)}")
            # else:
            
          if 1:
            f.write(
              f"{i}\t{str(predicted_spans)}\t{str(predicted_spans_I)}\t{str(predicted_spans_E)}"
              f"\t{p_B}\t{p_I}\t{p_E}"
              "\n")


elif "token_4cls" in eval_config.model_name:
  print("Token 4cls eval running...") 
  # Set prediction args
  # args.do_predict = True
  args = TrainingArguments(output_dir="./out", **eval_config.args)
  trainer = Trainer(
    model=model,
    args=args,
  )
  if eval_config.with_ground:
    # for key in tokenized_train_dataset.keys():
    for key in ['validation']:
      temp_offset_mapping = tokenized_train_dataset[key]["offset_mapping"]
      predictions = trainer.predict(tokenized_train_dataset[key])
      temp_untokenized_spans = untokenized_train_dataset[key]["spans"]
      # print(untokenized_train_dataset[key])

      preds = predictions.predictions
      # import ipdb; ipdb.set_trace()
      preds_s = torch.softmax(torch.from_numpy(preds).float(), dim=2)
      preds_proba = np.max(preds_s.numpy(), axis=2)

      preds = np.argmax(preds, axis=2)
      # for thresh in np.linspace(0.7, 0.95, 10): # Best is 0.75
      for thresh in [0.8]: 
        f1_scores = []
        edit_scores = []
        with open(
          # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
          os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
        ) as f:
          # f.write(f"Model Name: {suffix}, Dataset: {key}\n")
          f.write(f"Model Name: {suffix}, Dataset: {key}, Threshold: {thresh}\n")

        # outputs = []
        # gts = [] 
        
        with open(
          # os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "w"
          os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "w"
        ) as f:
          for i, (pred, pred_proba) in tqdm(enumerate(zip(preds, preds_proba))):
            # print(key,i)
            ## Batch Wise
            # print(len(prediction))
            predicted_spans = []
            predicted_spans_I = []
            predicted_spans_E = []
            for j, (tokenwise_prediction, token_p) in enumerate(
              zip(
              pred[: len(temp_offset_mapping[i])],
              pred_proba[: len(temp_offset_mapping[i])]
              )
            ):
              # thresh = 0.8
              if token_p < thresh: 
                continue
              if tokenwise_prediction == 1:
                predicted_spans += list(
                  range(
                    temp_offset_mapping[i][j][0],
                    temp_offset_mapping[i][j][1],
                  )
                )
              elif tokenwise_prediction == 2:
                predicted_spans_I += list(
                  range(
                    temp_offset_mapping[i][j][0],
                    temp_offset_mapping[i][j][1],
                  )
                )
              elif tokenwise_prediction == 3:
                predicted_spans_E += list(
                  range(
                    temp_offset_mapping[i][j][0],
                    temp_offset_mapping[i][j][1],
                  )
                )
            # if i == len(preds) - 1:
            #     f.write(f"{i}\t{str(predicted_spans)}\t{str(predicted_spans)}")
            # else:
            if 1:
              f.write(f"{i}\t{str(predicted_spans)}\t{str(predicted_spans_I)}\t{str(predicted_spans_E)}\n")
            
            # Edit distance
            ranges = get_ranges(predicted_spans)
            ranges_I = get_ranges(predicted_spans_I)
            ranges_E = get_ranges(predicted_spans_E)
            # print("ranges: ", ranges)
            # Get substring from span range
            range_merged = []
            for i, rng_I in enumerate(ranges_I):
              if rng_I is None:
                continue
              for j, rng_B in enumerate(ranges):
                if rng_B is None:
                  continue

                # print(rng_B, rng_I)
                if (rng_I.start == (rng_B.stop + 2)) and (rng_B.stop + 1 not in predicted_spans): 
                  # handle one space between them
                  new_rng = range(rng_B.start, rng_I.stop)
                  # print(new_rng)
                  range_merged.append(new_rng)
                  ranges_I[i] = None
                  ranges[j] = None
                  # exit()
                  break
            
            # Filter all Nones from ranges and ranges_I
            ranges = [rng for rng in ranges if rng is not None]
            ranges_I = [rng for rng in ranges_I if rng is not None]

            ranges = ranges + ranges_I + range_merged + ranges_E
            # print("ranges: ", ranges)
            # Sort ranges by start
            ranges.sort(key=lambda x: x.start)
              
            sentence = untokenized_train_dataset[key]["sentence"][i]
            gt       = untokenized_train_dataset[key]["gt"][i]
            output = ""
            prev_s = 0
            for i, span in enumerate(ranges):
              # if type(span) == int:
              #   s = span
              #   e = span + 1
              # else:
              if 1:
                s = span.start
                e = span.stop + 1
              
              if s >= len(sentence):
                break

              if s in predicted_spans_E:
                output += sentence[prev_s:e] + "$$"
              else:
                output += sentence[prev_s:s] + "$" + sentence[s:e] + "$"
              prev_s = e
              
            if prev_s < len(sentence):
              output += sentence[prev_s:]
            
            output = replace_fixed(output)
            output = replace_end(output)
            
            # print("output: ", output)
            # gt = untokenized_train_dataset[key]["gt"][i]
            # save output and gt to file
            # with open(os.path.join(eval_config.save_dir, f"output_{key}_{suffix}.txt"), "a+") as outfile:
            #     outfile.write(gt + "\n")
            #     # outfile.write(sentence + "\n")
            #     outfile.write(output + "\n")
            #     outfile.write("\n")

            edit = edit_distance(output, gt)

            edit_scores.append(edit)
            # outputs.append(output)
            # gts.append(gt)

            f1_scores.append(
              f1(
                predicted_spans,
                eval(temp_untokenized_spans[i]),
              )
            )
          
        # from pandarallel import pandarallel

        # pandarallel.initialize()
        # tqdm.pandas()
        
        # df = pd.DataFrame({"output": outputs, "gt": gts})
        # # Computer edit distance
        # edit_scores = df.parallel_appy(lambda x: edit_distance(x["output"], x["gt"]), axis=1)

        with open(
          # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
          os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
        ) as f:
          f.write(str(np.mean(f1_scores)))
          f.write(" ")
          f.write(str(np.mean(edit_scores)))
          f.write("\n")
  else:
    for key in tokenized_test_dataset.keys():
      temp_offset_mapping = tokenized_test_dataset[key]["offset_mapping"]
      predictions = trainer.predict(tokenized_test_dataset[key])
      preds = predictions.predictions
      
      # Find the probability of the predicted label
      # import ipdb; ipdb.set_trace()
      preds_s = torch.softmax(torch.from_numpy(preds).float(), dim=2)
      preds_proba = np.max(preds_s.numpy(), axis=2)

      preds = np.argmax(preds, axis=2)
      f1_scores = []
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "a"
      ) as f:
        for i, (pred, pred_proba) in tqdm(enumerate(zip(preds, preds_proba))):
          # print(key,i)
          ## Batch Wise
          # print(len(prediction))
          predicted_spans = []
          predicted_spans_I = []
          predicted_spans_E = []
          p_B, p_I, p_E = [], [], []    
          for j, (tokenwise_prediction, token_p) in enumerate(
            zip(
            pred[: len(temp_offset_mapping[i])],
            pred_proba[: len(temp_offset_mapping[i])]
            )
          ):
            thresh = 0.8
            if token_p < thresh: 
              continue

            test = temp_offset_mapping[i][j][1] != 0
            # Repeat token range time
            token_p = np.repeat(token_p, temp_offset_mapping[i][j][1] - temp_offset_mapping[i][j][0])
            token_p = token_p.tolist()


            if tokenwise_prediction == 1:
              # import ipdb; ipdb.set_trace()
              if test: p_B += token_p

              predicted_spans += list(
                range(
                  temp_offset_mapping[i][j][0],
                  temp_offset_mapping[i][j][1],
                )
              )
            elif tokenwise_prediction == 2:
              if test: p_I += token_p
              predicted_spans_I += list(
                range(
                  temp_offset_mapping[i][j][0],
                  temp_offset_mapping[i][j][1],
                )
              )
            elif tokenwise_prediction == 3:
              if test: p_E += token_p
              predicted_spans_E += list(
                range(
                  temp_offset_mapping[i][j][0],
                  temp_offset_mapping[i][j][1],
                )
              )

          # if i == len(preds) - 1:
          #     f.write(f"{i}\t{str(predicted_spans)}")
          # else:
          #     f.write(f"{i}\t{str(predicted_spans)}\n")
          if 1:
            f.write(
              f"{i}\t{str(predicted_spans)}\t{str(predicted_spans_I)}\t{str(predicted_spans_E)}"
              f"\t{p_B}\t{p_I}\t{p_E}"
              "\n")


elif "token" in eval_config.model_name:
  print("Token 2cls eval running...") 
  args = TrainingArguments(output_dir="./out", **eval_config.args)
  trainer = Trainer(
    model=model,
    args=args,
  )
  if eval_config.with_ground:
    for key in tokenized_train_dataset.keys():
      temp_offset_mapping = tokenized_train_dataset[key]["offset_mapping"]
      predictions = trainer.predict(tokenized_train_dataset[key])
      temp_untokenized_spans = untokenized_train_dataset[key]["spans"]
      # print(untokenized_train_dataset[key])

      preds = predictions.predictions
      # import ipdb; ipdb.set_trace()
      preds_s = torch.softmax(torch.from_numpy(preds).float(), dim=2)
      preds_proba = np.max(preds_s.numpy(), axis=2)
      preds = np.argmax(preds, axis=2)
      f1_scores = []
      edit_scores = []
      with open(
        # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
      ) as f:
        f.write(f"Model Name: {suffix}, Dataset: {key}\n")
      
      with open(
        # os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "w"
        os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "w"
      ) as f:
        for i, (pred, pred_proba) in tqdm(enumerate(zip(preds, preds_proba))):
          # print(key,i)
          ## Batch Wise
          # print(len(prediction))
          predicted_spans = []
          for j, (tokenwise_prediction, token_p) in enumerate(
            zip(
            pred[: len(temp_offset_mapping[i])],
            pred_proba[: len(temp_offset_mapping[i])]
            )
          ):
            thresh = 0.8
            if token_p < thresh: 
              continue
            if tokenwise_prediction == 1:
              predicted_spans += list(
                range(
                  temp_offset_mapping[i][j][0],
                  temp_offset_mapping[i][j][1],
                )
              )
          if i == len(preds) - 1:
            f.write(f"{i}\t{str(predicted_spans)}")
          else:
            f.write(f"{i}\t{str(predicted_spans)}\n")
          
          # Edit distance
          ranges = get_ranges(predicted_spans)
          # print("ranges: ", ranges)
          # Get substring from span range
          sentence = untokenized_train_dataset[key]["sentence"][i]
          gt       = untokenized_train_dataset[key]["gt"][i]
          output = ""
          prev_s = 0
          for i, span in enumerate(ranges):
            if type(span) == int:
              s = span
              e = span + 1
            else:
              s = span.start
              e = span.stop + 1
            
            output += sentence[prev_s:s] + "$" + sentence[s:e] + "$"
            prev_s = e
            
          if prev_s < len(sentence):
            output += sentence[prev_s:]
          
          output = replace_fixed(output)
          output = replace_end(output)
          
          # print("output: ", output)
          # gt = untokenized_train_dataset[key]["gt"][i]
          # save output and gt to file
          # with open(os.path.join(eval_config.save_dir, f"output_{key}_{suffix}.txt"), "a+") as outfile:
          #     outfile.write(gt + "\n")
          #     # outfile.write(sentence + "\n")
          #     outfile.write(output + "\n")
          #     outfile.write("\n")

          edit = edit_distance(output, gt)

          edit_scores.append(edit)

          f1_scores.append(
            f1(
              predicted_spans,
              eval(temp_untokenized_spans[i]),
            )
          )
      with open(
        # os.path.join(eval_config.save_dir, f"eval_scores_{key}_{suffix}.txt"), "w"
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "a"
      ) as f:
        f.write(str(np.mean(f1_scores)))
        f.write(" ")
        f.write(str(np.mean(edit_scores)))
        f.write("\n")
  else:
    for key in tokenized_test_dataset.keys():
      temp_offset_mapping = tokenized_test_dataset[key]["offset_mapping"]
      predictions = trainer.predict(tokenized_test_dataset[key])
      preds = predictions.predictions
      preds_s = torch.softmax(torch.from_numpy(preds).float(), dim=2)
      preds_proba = np.max(preds_s.numpy(), axis=2)
      preds = np.argmax(preds, axis=2)
      f1_scores = []
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred_{key}_{suffix}.txt"), "a"
      ) as f:
        for i, (pred, pred_proba) in tqdm(enumerate(zip(preds, preds_proba))):
          # print(key,i)
          ## Batch Wise
          # print(len(prediction))
          predicted_spans = []
          for j, (tokenwise_prediction, token_p) in enumerate(
            zip(
            pred[: len(temp_offset_mapping[i])],
            pred_proba[: len(temp_offset_mapping[i])]
            )
          ):
            thresh = 0.8
            if token_p < thresh: 
              continue
            if tokenwise_prediction == 1:
              predicted_spans += list(
                range(
                  temp_offset_mapping[i][j][0],
                  temp_offset_mapping[i][j][1],
                )
              )
          if i == len(preds) - 1:
            f.write(f"{i}\t{str(predicted_spans)}")
          else:
            f.write(f"{i}\t{str(predicted_spans)}\n")


else:
  # QA Eval
  topk = eval_config.topk
  if os.path.exists(os.path.join(eval_config.save_dir, f"thresh.txt")):
    with open(os.path.join(eval_config.save_dir, f"thresh.txt")) as f:
      best_threshold = float(f.read().split("\n")[0])
  else:
    val_original = untokenized_train_dataset["validation"]

    all_predicted_spans = []
    best_threshold = -1
    best_macro_f1 = -1
    print("here")
    for row_number in tqdm(range(len(val_original))):
      row = val_original[row_number]
      context = row["text"]
      question = "ভুল"
      while True and topk > 0:
        try:
          if topk == 1:
            spans = [nlp(question=question, context=context, topk=topk)]
          else:
            spans = nlp(question=question, context=context, topk=topk)
          break
        except:
          topk -= 1
          if topk == 0:
            break
      all_predicted_spans.append(spans)  # [examples,topk]
    thresholds = np.linspace(0, 1, 100)
    for threshold in tqdm(thresholds):
      macro_f1 = 0
      for row_number in range(len(val_original)):
        row = val_original[row_number]
        ground_spans = eval(row["spans"])
        predicted_spans = all_predicted_spans[row_number]
        predicted_spans = [
          span
          for span in predicted_spans
          if torch.sigmoid(torch.tensor(span["score"])) > threshold
        ]

        final_predicted_spans = []
        for span in predicted_spans:
          final_predicted_spans += list(range(span["start"], span["end"]))

        final_predicted_spans = sorted(final_predicted_spans)
        macro_f1 += f1(final_predicted_spans, ground_spans)
      avg = macro_f1 / len(val_original)
      if avg > best_macro_f1:
        best_macro_f1 = avg
        best_threshold = threshold

    with open(os.path.join(eval_config.save_dir, f"thresh.txt"), "w") as f:
      f.write(str(best_threshold) + "\n")
      f.write(str(best_macro_f1))
  if eval_config.with_ground:
    for key in untokenized_train_dataset.keys():
      f1_scores = []
      temp_test_dataset = untokenized_train_dataset[key]
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
      ) as f:
        for row_number in tqdm(range(len(temp_test_dataset))):
          row = temp_test_dataset[row_number]
          context = row["text"]
          question = "ভুল"
          while True and topk > 0:
            try:
              if topk == 1:
                spans = [
                  nlp(question=question, context=context, topk=topk)
                ]
              else:
                spans = nlp(
                  question=question, context=context, topk=topk
                )
              break
            except:
              topk -= 1
              if topk == 0:
                break
          predicted_spans = spans
          predicted_spans = [
            span
            for span in predicted_spans
            if torch.sigmoid(torch.tensor(span["score"])) > best_threshold
          ]

          final_predicted_spans = []
          for span in predicted_spans:
            final_predicted_spans += list(range(span["start"], span["end"]))

          final_predicted_spans = sorted(final_predicted_spans)
          if row_number != len(temp_test_dataset) - 1:
            f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
          else:
            f.write(f"{row_number}\t{str(final_predicted_spans)}")
          f1_scores.append(f1(final_predicted_spans, eval(row["spans"])))
      with open(
        os.path.join(eval_config.save_dir, f"eval_scores_{key}.txt"), "w"
      ) as f:
        f.write(str(np.mean(f1_scores)))

  else:
    for key in untokenized_test_dataset.keys():
      temp_test_dataset = untokenized_test_dataset[key]
      with open(
        os.path.join(eval_config.save_dir, f"spans-pred-{key}.txt"), "w"
      ) as f:
        for row_number in range(len(temp_test_dataset)):
          row = temp_test_dataset[row_number]
          context = row["text"]
          question = "ভুল"
          while True and topk > 0:
            try:
              if topk == 1:
                spans = [
                  nlp(question=question, context=context, topk=topk)
                ]
              else:
                spans = nlp(
                  question=question, context=context, topk=topk
                )
              break
            except:
              topk -= 1
              if topk == 0:
                break
          predicted_spans = spans
          predicted_spans = [
            span
            for span in predicted_spans
            if torch.sigmoid(torch.tensor(span["score"])) > best_threshold
          ]

          final_predicted_spans = []
          for span in predicted_spans:
            final_predicted_spans += list(range(span["start"], span["end"]))

          final_predicted_spans = sorted(final_predicted_spans)
          if row_number != len(temp_test_dataset) - 1:
            f.write(f"{row_number}\t{str(final_predicted_spans)}\n")
          else:
            f.write(f"{row_number}\t{str(final_predicted_spans)}")
