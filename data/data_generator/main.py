import pandas as pd


class ErrorLayer:
    
    def gen_error(self,sentence_list, error_list):
        """
        sentence_list: list of words
        error_list: list of tuple( position_in_sentence list, 
        number of word in sentence_list, errored_words)
        
        returns : new error_list
        """
        pass

def add_error(s_list):
    print("sentence_list: ", s_list)
    print("len(s_list[0]): ", len(s_list[0]))
    for c in s_list[0]:
        print("c: ", c)
    pass

if __name__ == '__main__':
    csv_file = '../../../archive/data_v2/data_v2_processed_500.csv'
    correct_sentences = pd.read_csv(csv_file)
    print(correct_sentences.head(10))
    tot=  0
    for row in correct_sentences.iterrows():
        lst = row[1]['correct_sentence'].split(' ')
        add_error(lst)
        tot+=1
        if tot>10:
            break
    