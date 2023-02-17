# Lint as: python3
"""Example tagging for Toxic Spans based on Spacy.
 
Requires:
  pip install spacy sklearn
 
Install models:
  python -m spacy download en_core_web_sm
 
"""

import ast
import csv
import random
import statistics
import sys

import sklearn
import spacy

# sys.path.append("./evaluation")
from evaluation import semeval2021
from evaluation import fix_spans


def spans_to_ents(doc, spans, label):
    """Converts span indicies into spacy entity labels."""
    started = False
    left, right, ents = 0, 0, []
    for x in doc:
        if x.pos_ == "SPACE":
            continue
        if spans.intersection(set(range(x.idx, x.idx + len(x.text)))):
            if not started:
                left, started = x.idx, True
            right = x.idx + len(x.text)
        elif started:
            ents.append((left, right, label))
            started = False
    if started:
        ents.append((left, right, label))
    return ents


def read_datafile(filename, test=False):
    """Reads csv file with python span list and text."""
    data = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        count = 0
        for row in reader:
            if test:
                data.append(row["text"])

            else:
                fixed = fix_spans.fix_spans(ast.literal_eval(row["spans"]), row["text"])
                data.append((fixed, row["text"]))
    return data


def main():
    """Train and eval a spacy named entity tagger for toxic spans."""
    # Read training data
    print("loading training data")
    datasets = {}
    # datasets["clean_train"] = read_datafile("../data/clean_train.csv")

    # Read trial data for test.
    # print("loading dev data")
    # dev = read_datafile("../data/tsd_trial.csv")

    # train = train + dev

    # datasets["clean_trial"] = read_datafile("../data/clean_trial.csv")
    datasets["tsd_train"] = read_datafile("../data/tsd_train.csv")
    datasets["tsd_trial"] = read_datafile("../data/tsd_trial.csv")
    datasets["tsd_test"] = read_datafile("../data/tsd_test_spans.csv")

    # Convert training data to Spacy Entities
    nlp = spacy.load("en_core_web_sm")

    print("preparing training data")
    training_data = []
    for n, (spans, text) in enumerate(datasets["tsd_train"]):
        doc = nlp(text)
        ents = spans_to_ents(doc, set(spans), "TOXIC")
        training_data.append((doc.text, {"entities": ents}))

    toxic_tagging = spacy.blank("en")
    toxic_tagging.vocab.strings.add("TOXIC")
    ner = nlp.create_pipe("ner")
    toxic_tagging.add_pipe(ner, last=True)
    ner.add_label("TOXIC")

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [
        pipe for pipe in toxic_tagging.pipe_names if pipe not in pipe_exceptions
    ]

    print("training")
    with toxic_tagging.disable_pipes(*unaffected_pipes):
        toxic_tagging.begin_training()
        for iteration in range(30):
            random.shuffle(training_data)
            losses = {}
            batches = spacy.util.minibatch(
                training_data, size=spacy.util.compounding(4.0, 32.0, 1.001)
            )
            for batch in batches:
                texts, annotations = zip(*batch)
                toxic_tagging.update(texts, annotations, drop=0.5, losses=losses)
            print("Losses", losses)

    # Score on dev data.
    print("evaluation")
    for dataset in datasets.keys():
        scores = []
        with open(f"spans-pred-{dataset}.txt", "w") as f:
            for i, (spans, text) in enumerate(datasets[dataset]):
                pred_spans = []
                doc = toxic_tagging(text)
                for ent in doc.ents:
                    pred_spans.extend(
                        range(ent.start_char, ent.start_char + len(ent.text))
                    )
                score = semeval2021.f1(pred_spans, spans)
                f.write(f"{i}\t{str(pred_spans)}\n")
                scores.append(score)

        with open(f"eval_scores_{dataset}.txt", "w") as f:
            f.write(str(statistics.mean(scores)))


## Score : 0.615544

if __name__ == "__main__":
    main()