import pandas as pd
from split_error_layer import SplitErrorLayer
from merge_error_layer import MergeErrorLayer
from punctuation_error_layer import PunctuationErrorLayer
from transiterate_layer import TransiterateLayer
from spelling_error_layer import SpellingErrorLayer
import nltk
nltk.download('punkt', quiet=True)
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
tqdm.pandas()

class ErrorLayer:
    
    def gen_error(self,sentence_list, error_list):
        """
        sentence_list: list of words
        error_list: list of tuple( position_in_sentence list, 
        number of word in sentence_list, errored_words)
        
        returns : new error_list
        """
        pass

class ErrorGenerator:
    def __init__(self,error_prob = 0.40):
        self.error_prob = error_prob
        self.layers = []
        self.layers += [SplitErrorLayer(error_prob_in_sentence=0.5)]
        self.layers += [MergeErrorLayer(error_prob_in_sentence=0.5)]
        self.layers += [PunctuationErrorLayer(
        	error_prob_in_sentence = 0.6,
        	replace_prob=0.33,
        	remove_prob=0.33,
        	insert_prob=0.33
        )]
        self.layers += [TransiterateLayer(error_prob_in_sentence=0.5)]
        self.layers += [SpellingErrorLayer(error_prob_in_sentence=0.1)]
    
    def get_row(self, s_list,error_list):
        # pass
        correct_sentence = ' '.join(s_list)
        correct_sentence+='।'
        
        gt = ''
        sentence = ''
        correction = [] # list of tuple ((start_pos, end_pos), correction))
        pos_sentence = 0
        n=len(s_list)
        m=len(error_list)
        i=0
        j=0
        
        while i<n:
            if j<m and error_list[j][0] == i:
                gt+= '$'+error_list[j][2]+'$'
                
                start_pos = pos_sentence
                pos_sentence+=len(error_list[j][2])
                sentence+=error_list[j][2]
                end_pos = pos_sentence-1
                
                correct_words = ' '.join(s_list[i:i+error_list[j][1]])
                correction.append(((start_pos,end_pos),correct_words))
                i+=error_list[j][1]
                j+=1
            else:
                gt+=s_list[i]
                sentence+=s_list[i]
                pos_sentence+=len(s_list[i])
                i+=1
            if i<n:
                gt+=' '
                sentence+=' '
                pos_sentence+=1
        
        sentence+='।'
        gt+='।'
        
        # print("correct_sentence: ",correct_sentence)
        # print("sentence: ",sentence)
        # print("gt: ",gt)
        # print("correction: ",correction)
        return (correct_sentence, gt, sentence, correction)
    
    def gen_error(self,s_list):
        error_list = []
        if np.random.rand() < self.error_prob:
            for layer in self.layers:
                error_list = layer.gen_error(s_list, error_list)
        # print("s_list: ",s_list)
        # print("error_list: ",error_list)
        # print("--------")
        return self.get_row(s_list,error_list)
        

if __name__ == '__main__':
    np.random.seed(0)
    csv_file = '../../../archive/data_v2/data_v2_processed_500.csv'
    out_file = './tmp.csv'
    correct_sentences = pd.read_csv(csv_file)
    g = ErrorGenerator()
    tot=  0
    
    df = pd.DataFrame()
    df["correct_sentence"],df["gt"],df["sentence"],df["correction"] = \
                zip(*correct_sentences['correct_sentence'].progress_apply(lambda x: g.gen_error(nltk.word_tokenize(x))))
    
    count_correction = df['correction'].apply(lambda x: len(x))
    print(count_correction)
    counts = count_correction.value_counts().sort_index()
    ax=counts.plot(kind='bar')
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha='center', va='bottom')
    plt.show()
    print("counts: \n", counts)
    percents = counts/len(df)
    ax = percents.plot(kind='bar')
    for i, v in enumerate(percents):
        ax.text(i, v, f'{v:0.2f}', ha='center', va='bottom')
    plt.show()
    print("percents: \n", percents)
    df.to_csv(out_file, index=False)
    # for row in correct_sentences.iterrows():
    #     sentence = row[1]['correct_sentence']
    #     # split sentence into words and pancuation
    #     lst = nltk.word_tokenize(sentence)
    #     g.gen_error(lst)
    #     tot+=1
    #     if tot>10:
    #         break
    