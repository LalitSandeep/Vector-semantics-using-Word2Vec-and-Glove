import requests
from gensim.models import KeyedVectors
from gensim.models import word2vec
import logging
 
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
 

#-------- Loading the test data------------------
with open('File.txt') as f:    
    lines = f.read().splitlines()
    
#-----------Building the model using train data------------------------
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True,limit=100000)



#-------------- Splitting into words----------------------
data=[]
for word in lines:
    data.append(word.split(' '))
    
    
    

#------------------Making a dictionary of values with the eight analogy tasks as keys.

ar={}
for i in data:
    if(i[0]==':'):
        key=i[1]
        ar[key]=[]
    else:
        ar[key].append(i)

 
categ = ['capital-world', 'currency', 'city-in-state', 'family','gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative','gram6-nationality-adjective']
arr = {}
for classes in categ:
    arr[classes] = ar[classes]    

#----------------- Calculating accuracy by checking the analogy with 4 different words in a test instance.
 #----------------  Then we check the test data with our trained model i.e. whether the model is able to predict the analogy correctly or not.
print(categ)
   
for i in categ:  
    print(i)
    count=0
    for sentence in arr[i]:
        c=0
        for j in range(3): 
            if sentence[j] in model:                
                c=c+1        
        if c==3:            
            l=model.most_similar(positive=[sentence[1], sentence[2]], negative=[sentence[0]], topn=1)
            if l[0][0]==sentence[3]:
                count=count+1                
    accuracy=count/len(arr[i])
    print(accuracy*100)
    
    
    
    
#----------- Analogy test---------------------               
model.most_similar('enter',topn=10)
model.most_similar('increase',topn=10)






#------- We check 1 analogy test with "category : example" model.


# =============================================================================

       #1     emotion : grief :: furniture : couch        

result66='couch'
  
result6 = model.most_similar(positive=['grief', 'furniture'], negative=['emotion'], topn=1)
print(result6)  

if result6[0][0]==result66:
    acc=100
else:
    acc=0
    
print(acc)
        


# =============================================================================
   #2     beverage : milk : : animal : cat

result11='cat'
  
result = model.most_similar(positive=['milk', 'animal'], negative=['beverage'], topn=1)
print(result)  

if result[0][0]==result11:
    acc=100
else:
    acc=0
    
print(acc)
