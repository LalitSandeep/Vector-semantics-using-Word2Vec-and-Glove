import requests
from gensim.models import KeyedVectors
from gensim.models import word2vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
 
import logging
 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 
sentences = word2vec.Text8Corpus('GoogleNews-vectors-negative300.bin') 

with open('File.txt') as f:
    lines = f.read().splitlines()
    


glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)





filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)
result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(result)

data=[]
for word in lines:
    data.append(word.split(' '))
    



dfdic={}
for i in data:
    if(i[0]==':'):
        key=i[1]
        dfdic[key]=[]
    else:
        dfdic[key].append(i)

 
reqd = ['capital-world', 'currency', 'city-in-state', 'family','gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative','gram6-nationality-adjective']
dfdict = {}
for classes in reqd:
    dfdict[classes] = dfdic[classes]    



for i in reqd:  
    print(i)
    count=0
    for sentence in dfdict[i]:
        c=0
        for j in range(3): 
            if sentence[j] in model:                
                c=c+1        
        if c==3:            
            l=model.most_similar(positive=[sentence[1], sentence[2]], negative=[sentence[0]], topn=1)
            if l[0][0]==sentence[3]:
                count=count+1                
    accuracy=count/len(dfdict[i])
    print(accuracy*100)
    
     
    