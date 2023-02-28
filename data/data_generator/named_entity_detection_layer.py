from bnlp import NER

class NamedEntityDetectionLayer:
    def __init__(self) -> None:
        self.model_path = "bn_ner.pkl"
        self.bn_ner = NER()
        pass	
    
    def get_tag(self,sentence):
        ret_list = []
        res = self.bn_ner.tag(self.model_path, sentence)
        
        s_list = [x[0] for x in res]
        # print("res: ",res)
        i = 0
        for x in res:
            if x[1] != 'O':
                ret_list.append((i,1,x[0]))
            # else:
                # print("x: ",x)
            i+=1
        
        return s_list,ret_list