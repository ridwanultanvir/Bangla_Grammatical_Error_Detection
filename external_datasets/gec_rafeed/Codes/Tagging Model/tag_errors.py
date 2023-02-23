import pandas as pd
import errant
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import argparse

annotator = errant.load('en')
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

prefixes = ['M:', 'R:', 'U:']

def get_error_tags(orig_and_cor):
    tags = []
    orig = annotator.parse(orig_and_cor[0])
    cor = annotator.parse(orig_and_cor[1])
    edits = annotator.annotate(orig, cor)
    
    for e in edits:
        tags.append(e.type)
    
    return tags

def encode_tag(error_tags):
    label = []

    for c in classes:
        match = 0
        for tag in error_tags:
            if tag[:2] in prefixes:
                tag = tag[2:]
            if tag == c:
                match = 1
        label.append(match)

    return label

def main(args):
    df = pd.read_csv(args.data_path, index_col=0)
    print(f'Read {len(df)} entries from {args.data_path}')
    df = df.dropna()
    df.original = df.original.astype('string')
    df.corrected = df.corrected.astype('string')

    with Pool(processes=args.n_workers) as pool:
        print(f'Generating errant tags for {len(df)} entries:')
        error_tags = list(tqdm(pool.imap(get_error_tags, df[['original', 'corrected']].to_numpy(), chunksize=args.chunksize), total=len(df)))

        print(f'Generating text labels for {len(error_tags)} sentences:')
        labels = list(tqdm(pool.imap(encode_tag, error_tags, chunksize=args.chunksize), total=len(error_tags)))


    labeled_df = pd.DataFrame(zip(df.corrected, labels), columns=['text', 'label'])

    print(f'Saving tagged data to {args.save_path}')
    labeled_df.to_parquet(args.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to csv data file', type=str)
    parser.add_argument('save_path', help='Path where processed data should be saved', type=str)
    parser.add_argument('--n_workers', help='Number of processes to use (default: 10)', default=10, type=int)
    parser.add_argument('--chunksize', help='Number of entries to process at one time by each worker (default: 100)', default=100, type=int)
    args = parser.parse_args()

    main(args)

