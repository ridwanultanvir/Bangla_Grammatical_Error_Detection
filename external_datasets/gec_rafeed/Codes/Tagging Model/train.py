
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import DatasetDict, load_dataset, load_metric, ClassLabel, Sequence, Value
import torch
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
import argparse
import wandb
from tqdm import tqdm
import pandas as pd

classes = ['ADJ', 
           'ADJ:FORM', 
           'ADV', 
           'CONJ', 
           'CONTR', 
           'DET', 
           'MORPH', 
           'NOUN', 
           'NOUN:INFL',
           'NOUN:NUM', 
           'NOUN:POSS', 
           'ORTH', 
           'OTHER', 
           'PART', 
           'PREP', 
           'PRON', 
           'PUNCT', 
           'SPELL', 
           'VERB', 
           'VERB:FORM', 
           'VERB:INFL', 
           'VERB:SVA', 
           'VERB:TENSE',
           'WO']

id2label = {i:x for i,x in enumerate(classes)}
label2id = {x:i for i,x in enumerate(classes)}

def compute_metrics(eval_pred):
    threshold = 0.3
    logits, labels = eval_pred
    predictions = torch.sigmoid(torch.as_tensor(logits)).cpu()
    pred_labels = torch.where(predictions < threshold, 0, 1)
    f = fbeta_score(labels, pred_labels, beta=0.5, zero_division=0, average='weighted')
    precision = precision_score(labels, pred_labels, zero_division=0, average='weighted')
    recall = recall_score(labels, pred_labels, zero_division=0, average='weighted')
    return {'f0.5_score':f, 'precision':precision, 'recall':recall}

def get_predictions(trainer, dataset, save_path, threshold=0.3):
    print('Making predictions on given data')
    predictions = torch.sigmoid(torch.as_tensor(trainer.predict(dataset)[0]))
    pred_labels = torch.where(predictions < threshold, 0, 1)
    new_tags = np.apply_along_axis(lambda x: ''.join(['a' if not y else 'b' for y in x]), -1, pred_labels)

    print(f'\nconverting {len(dataset)} predictions to right format:')
    entries = [f'grammar_error: ({new_tags[i]}) {dataset[i]["text"]}' for i in tqdm(range(len(dataset)))]

    df = pd.DataFrame(entries, columns=['source'])
    print(f'\nsaving tagged data to {save_path}')
    df.to_csv(save_path)


def train(args):
    dataset = load_dataset('parquet', data_files={'train':args.train_data, 'eval':args.eval_data, 'test':args.test_data})
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    def tokenize_function(sample):
        return tokenizer(sample['text'], padding='max_length', truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['eval']
    test_dataset = tokenized_datasets['test']

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, id2label=id2label, label2id=label2id, problem_type='multi_label_classification', num_labels=len(classes)).to('cuda')
    training_args = TrainingArguments(output_dir='checkpoints', overwrite_output_dir=True, evaluation_strategy='steps', report_to='wandb', eval_steps=500, save_total_limit=15, learning_rate=args.lr)

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset, compute_metrics=compute_metrics)

    if not args.eval_only and not args.predict:
        trainer.train()

        if args.save_path is not None:
            model.save_pretrained(args.save_path)
            print(f'saved model to {args.save_path}')

    if args.eval_only:
        results = trainer.evaluate()
        print(results)
    
    if args.predict:
        get_predictions(trainer, test_dataset, args.prediction_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', help='path to training data in json format')
    parser.add_argument('--eval_data', help='path to eval data in json format')
    parser.add_argument('--test_data', help='path to test data in json format')
    parser.add_argument('--model_name', help='name of/path to model to load, either from huggingface or from local directory', default='roberta-base')
    parser.add_argument('--tokenizer', help='name of tokenizer to use', default='roberta-base')
    parser.add_argument('--save_path', help='where to save the model')
    parser.add_argument('--eval_only', help='only run evaluation', default=False)
    parser.add_argument('--predict', help='only predict/get outputs', default=False)
    parser.add_argument('--prediction_save_path', help='path to save predictions', default='')
    parser.add_argument('--lr', help='learning rate to start with (default: 5e-5)', default=5e-5, type=float)
    args = parser.parse_args()

    print(args)

    train(args)
