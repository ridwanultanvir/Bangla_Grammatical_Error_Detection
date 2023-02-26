import json
import pandas as pd
from tqdm import tqdm
import spacy
from multiprocessing import Pool
from spacy.tokenizer import Tokenizer
from argparse import ArgumentParser
nlp = spacy.load("en_core_web_md")
tokenizer = Tokenizer(nlp.vocab)

def process_line(line):
    sents = []
    obj = json.loads(line)['text']
    doc = nlp(obj)
    for sent in doc.sents:
        tokenized = nlp(sent.text.strip())
        tokens = [x.text for x in tokenized]
        sents.append(' '.join(tokens))
    return sents

def process_file(path, out_path):
    print(f'starting processing for {path}')
    sents  = []
    with open(path, 'r') as f:
        lines = f.readlines()
    with Pool(20) as pool:
        sents = list(tqdm(pool.imap(process_line, lines, chunksize=100), total=len(lines)))
    
    sents = [x for y in sents for x in y]
    df = pd.DataFrame(sents, columns=['text'])
    print(f'\nwriting {len(sents)} sentences to {out_path}')
    df.to_parquet(out_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_path', help='path to input json file to be tokenized & split into sentences', type=str)
    parser.add_argument('output_path', help='path to parquet file to output processed data to', type=str)
    args = parser.parse_args()

    process_file(args.input_path, args.output_path)
    # process_file('../data/c4/c4-train.00003-of-01024.json', '../data/c4/train-00003.parquet')
    # process_file('../data/c4/c4-train.00004-of-01024.json', '../data/c4/train-00004.parquet')
    # process_file('../data/c4/c4-train.00005-of-01024.json', '../data/c4/train-00005.parquet')