import numpy as np

class PunctuationErrorLayer():
    
    def __init__(self,replace_prob=0.33,remove_prob=0.33,insert_prob=0.33,error_prob_in_sentence=0.5) -> None:
        self.replace_prob = replace_prob
        self.remove_prob = remove_prob
        self.insert_prob = insert_prob
        self.error_prob_in_sentence = error_prob_in_sentence
        self.puncs=['।', '!', '?',',',':','-','(',')',':',';']
        pass

    def gen_error(self, s_list, error_list):
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
                if np.random.rand()<=self.error_prob_in_sentence:
                    if s_list[i] in self.puncs:
                        idx = np.random.randint(0,len(self.puncs))
                        if self.puncs[idx]!=s_list[i]:
                            # replace
                            if np.random.rand()<=self.replace_prob:
                                ret_error_list.append((i,1,self.puncs[idx]))
                            pass
                        else:
                            # remove
                            if np.random.rand()<=self.remove_prob:
                                ret_error_list.append((i,1,''))
                    else:
                        # insert
                        if np.random.rand()<=self.insert_prob:
                            idx = np.random.randint(0,len(self.puncs))
                            ret_error_list.append((i,0,self.puncs[idx]))
                        pass
                i+=1
        
        return ret_error_list