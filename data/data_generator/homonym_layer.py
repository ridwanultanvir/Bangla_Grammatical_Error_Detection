import pandas as pd
import numpy as np
class HomoynmLayer():
    
    def __init__(self) -> None:
        # frequnt_wordlist_file = './transiterate/en_bn.csv'
        # self.dict = pd.read_csv(frequnt_wordlist_file)
        # # print("dict head: ",self.dict.head(10))
        # self.dict.sort_values(by='bangla',
        #                       axis=0,
        #                       inplace=True,
        #                       ignore_index=True
        #                     )
        # self.dict.drop_duplicates(subset='bangla',
        #                           inplace=True,
        #                           ignore_index=True
        #                           )
        
        self.homoynm_dict = pd.read_csv('./homonyms2.csv')
        # self.homoynm_dict = homonym_df.to_dict()
        print("homonym dict:", self.homoynm_dict)
        # two columns : word,homonyms
        # homonyms is an array only take the first position of the array
        # self.phonetic_dict = {
        #     'ঋ' : ['রি'],
        #     'ঐ': ['অই'],
        #     'ঔ': ['অউ'],
        #     'খ': ['ক'],
        #     'ঙ': ['◌ং'],
        #     'ঝ': ['জ'],
        #     'ঠ' : ['ট','ত'],
        #     'ড' : ['দ'],
        #     'ঢ': ['ড'],
        #     'থ': ['ত','ট'],
        #     'দ' : ['ড'],
        #     'ধ' : ['দ','ড'],
        #     'ভ' : ['ব'],
        #     'য' : ['জ'],
        #     'ৎ' : ['ত'],
        #     'ং' : ['ঙ'],
        #     'ঃ': ['হ'],
        #     '‍ঁ' : [''],
        #     'ৈ'  :['ই'],
        #     'ৌ':['উ'],
        #     '‍ঢ়': ['র'],
        #     'ড়' : ['র'],
        #     'ৃ' : ['রি'],
        #     'অ' : ['ও'],
        #     'ই':['ঈ'],
        #     'উ' : ['ঊ'],
        #     'চ': ['ছ'],
        #     'ট' :['ত'],
        #     'ড': ['দ'],
        #     'ন': ['ণ'],
        #     'য' : ['জ'],
        #     'শ' : ['স','ষ'],
        #     'ি' : ['◌ী'],
        #     'ু' : ['◌ূ']
        # }
        
        pass
    
    def gen_word(self,word):
        ret=''
        for c in word:
            if c in self.homoynm_dict:
                ret+=self.homoynm_dict[c][np.random.randint(len(self.homoynm_dict[c]))]
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
                # print("to_search: ",to_search)
                idx = self.homoynm_dict['word'].searchsorted(to_search)
                # print("idx: ",idx," to_search: ",to_search)
                if idx<len(self.homoynm_dict['word']) and self.homoynm_dict['word'][idx]==to_search:
                    # print("to_search: ",to_search)
                    ret_error_list.append((i,1,self.homoynm_dict['homonyms'][idx]))
                # if np.random.sample()<0.2:
                #     ret_error_list.append((i,1,self.gen_word(to_search)))
                i+=1
        
        return ret_error_list