from bnlp import NER

class NamedEntityDetectionLayer:
    def __init__(self) -> None:
        self.model_path = "bn_pos.pkl"
        self.bn_ner = NER()
        pass	
    
    def gen_error(self,s_list,e_list):
        ret_e_list = []
        res = self.bn_ner.tag(self.model_path, s_list)
        # print("s_list: ",s_list)
        # print("res: ",res)
        assert len(s_list) == len(res)
        n=len(s_list)
        i=0
        while i<n:
            if res[i][1] != 'O':
                ret_e_list.append((i,1,s_list[i]))
            i+=1
        
        return ret_e_list