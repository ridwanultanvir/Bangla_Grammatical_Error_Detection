import pandas as pd
import numpy as np
class TransiterateLayer():
    
    def __init__(self) -> None:
        frequnt_wordlist_file = './transiterate/en_bn.csv'
        self.dict = pd.read_csv(frequnt_wordlist_file)
        # print("dict head: ",self.dict.head(10))
        self.dict.sort_values(by='bangla',
                              axis=0,
                              inplace=True,
                              ignore_index=True
                            )
        self.dict.drop_duplicates(subset='bangla',
                                  inplace=True,
                                  ignore_index=True
                                  )
    
    def gen_word(self,word):
        ret=''
        for c in word:
            if c in self.phonetic_dict:
                ret+=self.phonetic_dict[c][np.random.randint(len(self.phonetic_dict[c]))]
            else:
                ret+=c
        return ret
    
    
    def gen_error(self,s_list, error_list):
        n = len(s_list)
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
                to_search = s_list[i]
                idx = self.dict['bangla'].searchsorted(to_search)
                if idx<len(self.dict['bangla']) and self.dict['bangla'][idx]==to_search:
                    # print("to_search: ",to_search)
                    ret_error_list.append((i,1,self.dict['english'][idx]))
                i+=1
        
        return ret_error_list