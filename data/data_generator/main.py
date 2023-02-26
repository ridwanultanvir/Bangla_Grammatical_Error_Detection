import pandas as pd
from split_error_layer import SplitErrorLayer
from merge_error_layer import MergeErrorLayer

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
    def __init__(self):
        self.layers = [SplitErrorLayer()]
        self.layers += [MergeErrorLayer()]
    
    def gen_error(self,s_list):
        error_list = []
        for layer in self.layers:
            error_list = layer.gen_error(s_list, error_list)
        print("s_list: ",s_list)
        print("error_list: ",error_list)

if __name__ == '__main__':
    csv_file = '../../../archive/data_v2/data_v2_processed_500.csv'
    correct_sentences = pd.read_csv(csv_file)
    # print(correct_sentences.head(10))
    g = ErrorGenerator()
    # s_list = ['ট্রাম্প', 'তাঁর', 'রাজনীতির', 'জন্য', 'প্রধানত', 'ব্যবহার', 'করেন', 'উগ্র', 'জাতীয়তাবাদী', 'সুড়সুড়ি']
    # g.gen_error(s_list)
    tot=  0
    
    for row in correct_sentences.iterrows():
        lst = row[1]['correct_sentence'].split(' ')
        g.gen_error(lst)
        tot+=1
        if tot>100:
            break
    