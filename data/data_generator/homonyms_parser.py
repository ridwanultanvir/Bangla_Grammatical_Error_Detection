from shobdohash import ShobdoHash
import pandas
txt_file = './homonyms.txt'
f = open(txt_file, 'r')
lst = [line.strip() for line in f if line.strip()]
s = ShobdoHash()

from bnlp import NLTKTokenizer

bnltk = NLTKTokenizer()

graph = {}

def add_edge(w1,w2):
    if w1 not in graph:
        graph[w1] = set()
    graph[w1].add(w2)

print("number of lines: ",len(lst))
for i in range(0,len(lst),2):
    w1 = bnltk.word_tokenize(lst[i])[0]
    w2 = bnltk.word_tokenize(lst[i+1])[0]
    # print("w1: ",w1)
    # print("w1: ",w2)
    add_edge(w1,w2)
    add_edge(w2,w1)
    # if s(w1)!=s(w2):
    #     print("i: ",i)
    #     print("w1: ",w1)
    #     print("w2: ",w2)
    # print("")
assert len(lst)%2==0, "Number of lines in homonyms.txt should be even"
graph = [(k,list(v)) for k,v in graph.items()]
# for i,j in graph:
#     print("i: ",i,"j: ",j)
df = pandas.DataFrame(graph, columns=['word','homonyms'])
# print("head: \n",df.head())
df.to_csv('./homonyms.csv', index=False)