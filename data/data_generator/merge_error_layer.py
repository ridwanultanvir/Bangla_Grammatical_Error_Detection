import pandas as pd

class MergeErrorLayer:
    
    def __init__(self,**kwargs):
        
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
        while i+1<n:
            if j<m and i==error_list[j][0]:
                ret_error_list.append(error_list[j])
                i+=error_list[j][1]
                j+=1
            else:
                to_search = sentence_list[i]+sentence_list[i+1]
                idx = self.dict['word'].searchsorted(to_search)
                if (idx< len(self.dict['word'])) and \
                        (self.dict['word'][idx] == to_search):
                    ret_error_list.append((i,2,to_search))
                    i+=2
                else:
                    i+=1
        
        return ret_error_list

if __name__ == '__main__':
    pass


    