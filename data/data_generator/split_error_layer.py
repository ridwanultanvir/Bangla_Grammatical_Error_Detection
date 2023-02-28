import pandas as pd
import numpy as np
class SplitErrorLayer:
    
    def __init__(self,error_prob_in_sentence=0.5):
        self.error_prob_in_sentence = error_prob_in_sentence
        dict_file = './two_dictwords.csv'
        # laod dict
        self.dict = pd.read_csv(dict_file)
        # print(self.dict.head(10))
        self.dict['word'] = self.dict['word'].apply(lambda x: x.strip())
        self.dict.sort_values(by='word', 
                              axis=0,
                              inplace=True,
                              ignore_index=True
                            )
        
        # print(self.dict.head(10))
        self.dict.drop_duplicates(subset='word',
                                  inplace=True,
                                  ignore_index=True
                                  )
        # print(self.dict.head(10))
        
    
    def gen_error(self,sentence_list, error_list):
        n = len(sentence_list)
        m = len(error_list)
        i=0
        j=0
        ret_error_list = []
        while i<n:
            if j<m and i==error_list[j][0]:
                ret_error_list.append(error_list[j])
                i+=error_list[j][1]
                j+=1
            else:
                error_words = None
                word = sentence_list[i]
                ln = len(word)
                for pos in range(1,ln):
                    to_search = [word[:pos],word[pos:]]
                    idxs = self.dict['word'].searchsorted(to_search)
                    if idxs[0]>=len(self.dict['word']) or idxs[1]>=len(self.dict['word']):
                        continue
                    if (self.dict['word'][idxs] == to_search).all():
                        # print("to_search: ",to_search)
                        error_words = to_search
                        break
                if error_words and np.random.rand()<self.error_prob_in_sentence:
                    ret_error_list.append((i,1,' '.join(error_words)))
                
                i+=1
        
        return ret_error_list

if __name__ == '__main__':
    pass


    