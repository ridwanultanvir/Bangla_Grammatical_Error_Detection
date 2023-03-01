import numpy as np
import pandas as pd
class SpellingErrorLayer():
    
    def __init__(self,error_prob_in_sentence = 0.5) -> None:
        frequnt_wordlist_file = './en_bn.csv'
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
        self.error_prob_in_sentence = error_prob_in_sentence
        self.phonetic_dict = {
            'ঋ' : ['রি'],
            'ঐ': ['অই'],
            'ঔ': ['অউ'],
            'খ': ['ক'],
            'ঙ': ['◌ং'],
            'ঝ': ['জ'],
            'ঠ' : ['ট','ত'],
            'ড' : ['দ'],
            'ঢ': ['ড'],
            'থ': ['ত','ট'],
            'দ' : ['ড'],
            'ধ' : ['দ','ড'],
            'ভ' : ['ব'],
            'য' : ['জ'],
            'ৎ' : ['ত'],
            'ং' : ['ঙ'],
            'ঃ': ['হ'],
            '‍ঁ' : [''],
            'ৈ'  :['ই'],
            'ৌ':['উ'],
            '‍ঢ়': ['র'],
            'ড়' : ['র'],
            'ৃ' : ['রি'],
            'অ' : ['ও'],
            'ই':['ঈ'],
            'উ' : ['ঊ'],
            'চ': ['ছ'],
            'ট' :['ত'],
            'ড': ['দ'],
            'ন': ['ণ'],
            'য' : ['জ'],
            'শ' : ['স','ষ'],
            'ি' : ['◌ী'],
            'ু' : ['◌ূ']
        }
        
        pass
    
    def pure_char(self,s):
        ret=''
        for c in s:
            if ord(c) == 9676:
                continue
            ret+=c
        return ret
    def gen_word(self,word):
        ret=''
        assert len(word)>0
        # print("len(word) = ",len(word))
        total_error = np.random.randint(1,(len(word)+1)//2+1)
        for c in word:
            if c in self.phonetic_dict:
                if total_error == 0:
                    ret+=c
                else:
                    
                    ret+=self.pure_char(np.random.choice(self.phonetic_dict[c]))
                    total_error-=1
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
                    err_word = self.gen_word(to_search)
                    if err_word != to_search and np.random.sample()<self.error_prob_in_sentence:
                        ret_error_list.append((i,1,err_word))
                i+=1
        
        return ret_error_list