import numpy as np

class PunctuationErrorLayer():
    
    def __init__(self) -> None:
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
                if s_list[i] in self.puncs:
                    idx = np.random.randint(0,len(self.puncs))
                    if self.puncs[idx]!=s_list[i]:
                        # replace
                        ret_error_list.append((i,1,self.puncs[idx]))
                        pass
                    else:
                        # remove
                        ret_error_list.append((i,1,''))
                else:
                    # insert
                    # idx = np.random.randint(0,len(self.puncs))
                    # ret_error_list.append((i,0,self.puncs[idx]))
                    pass
                i+=1
        
        return ret_error_list