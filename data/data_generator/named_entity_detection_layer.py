from ner import NER
class NamedEntityDetectionLayer:
    def __init__(self) -> None:
        self.model_path = "./../../dcspell/model/bn_ner.pkl"
        self.bn_ner = NER(model_path=self.model_path)
        pass	
    
    def get_tag(self,sentence):
        ret_list = []
        res = self.bn_ner.tag (sentence)
        
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