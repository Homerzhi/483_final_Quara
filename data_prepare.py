
import numpy as np
from nltk import *
import nltk




'''
read train data
'''
#"id","qid1","qid2","question1","question2","is_duplicate"
inputfile='train.csv'


traindata=[]
print "reading data..."
with open(inputfile) as files:
    for line in files.readlines(): 
        if line.count(',')==5:    #if find more than 5 comma, skip this feature  
            line=line.replace("'"," ")
            line=line.replace("?"," ")
            line=line.replace("#"," ")
            line=repr(line).replace("\\"," ")
            line=line.lower()
            line=line.split('","') 
            line=line[-3:]        #only last three column matters
            line[-1]=line[-1][0]
            traindata.append(line)

x=len(traindata)




'''
find question type of each question
'''
questionType=('why','what','where','who','whom',\
'whose','when','can','could','may','might','if','will','would','has','have',\
'do','doe','did','am','is','was','were',)

qt=dict()
for i in range(len(questionType)):
    qt[questionType[i]]=i

qtype=[]                          #*******************************2
for i in range(len(traindata)):
    temp=[0]*2
    for t in questionType:
        if t in traindata[i][0]:
            temp[0]+=qt[t]
        if t in traindata[i][1]:
            temp[1]+=qt[t]
    qtype.append(temp)




'''
# of same words 
# of diff words in two question, 
'''
import copy


numsd=[]               #************************************2
traincopy=[]
for d in traindata:
    temp=[]
    q1,q2=d[0].split(),d[1].split()
    ns=0
    l=copy.copy(q1)        
    for n in l: #find the same number
        if n in q1 and n in q2:
            ns+=1
            q1.remove(n)
            q2.remove(n)
    nd=((len(q1)-len(q2)))**2
    dif=[]
    dif.append(q1)
    dif.append(q2)
    traincopy.append(dif)    
    temp.append(ns)
    temp.append(nd)
    numsd.append(temp)



'''
delete not important words from the traincopy which is after delete the same words list
'''
important=['VB', 'NN',  'MD', 'JJ','W'] 


def delete_nonimportant(l, flag):
    tem=[]
    for w in pos_tag(l):
        for i in important:
            if i in w[1]:
                tem.append(w[flag])                
    return tem

trainTag=[] 
for i in range(len(traincopy)):
    traincopy[i][0]=delete_nonimportant(traincopy[i][0],0)      #question 1, words
    traincopy[i][1]=delete_nonimportant(traincopy[i][1],0)      #question 2, words
    emp=[]
    emp.append(delete_nonimportant(traincopy[i][0],1) )     #question 1, tag
    emp.append(delete_nonimportant(traincopy[i][1],1) )     #question 2, tag
    emp.append(traincopy[i][-1])
    trainTag.append(emp)


'''
traincopy has the different words in two questions
find frequence of words in each question

from nltk.corpus import words
#'market' in words.words()  #check if english words

frequence=dict()
with open('en10000.txt') as files:
    count=0
    for line in files.readlines(): 
        frequence[line[:-1]]=count
        count+=1

freq=[]
for i in range(len(traincopy)):
    temp=[0]*2
    if traincopy[i][0]:
        for w in traincopy[i][0]:
            if w in words.words():
                temp[0]+=frequence[w]
    if traincopy[i][1]:
        for w in traincopy[i][1]:
            if w in words.words():
                temp[1]+=frequence[w]
    freq.append(temp)

'''


'''
find how many tags are the same
'''
samtags=[]  #***************************1

def find_same_tags(l1,l2):
    tem=0
    l=copy.copy(l1)
    for t in l:
        if t in l2:
            l1.remove(t)
            l2.remove(t)
            tem+=1
    return tem


for i in range(len(trainTag)):
     samtags.append(find_same_tags(trainTag[i][0],trainTag[i][1]))

'''
 compare difference 
'''
tagdiff=[]    #*********************************1

for i in range(len(trainTag)):
    tagdiff.append( (len(trainTag[i][0])- len(trainTag[i][1]))**2)





'''
synonym

from PyDictionary import PyDictionary
dictionary=PyDictionary()
import enchant
ench=enchant.Dict('en_US')

#ench.check('Life')   #check if english word
#dictionary.synonym("Life")

def synonyms(l1,l2):
    q=0
    if l1 and l2:
        for w1 in l1:
            if ench.check(w1):
                for w2 in l2:
                    if ench.check(w2):
                        if dictionary.synonym(w1) and w2 in dictionary.synonym(w1):
                            q+=1
    return q


synms=[]        #********************************
for i in range(len(train)):
    synms.append(synonyms(train[i][0],train[i][1]))

'''




 


'''
similarity of two sentence

from nltk.corpus import wordnet

def similaritys(l1,l2):
    sm = [0]
    if len(l1)>len(l2):
        l1,l2=l2,l1
    for word1 in l1:
        s=0.0
        for word2 in l2:
            w1 = wordnet.synsets(word1)
            w2 = wordnet.synsets(word2)
            if w1 and w2: 
                if w1[0].wup_similarity(w2[0]):
                    s+=w1[0].wup_similarity(w2[0])                
        sm.append(s)
    return sm

simi=[] #*****************************************
for i in range(len(traincopy)):
    simi.append(similaritys(traincopy[i][0],traincopy[i][1]))

for i in range(len(simi)):
    simi[i]=sum(simi[i])
'''


'''
columns:
q1_type, q2_type, #same_word, #diff_word, samtags, tagdiff, simi, target
'''
target=[]

for d in traindata:
    target.append(int(d[-1]))



traindata=np.zeros((x,6))
qtypem=np.reshape(qtype, (x,2))
numsdm=np.reshape(numsd,(x,2))
tagm=np.reshape(samtags, (x,1))
tagtm=np.reshape(tagdiff, (x,1))
#sim=np.reshape(simi, (x,1))
targets=np.reshape(target, (x,1))
targetsa=np.squeeze(targets)

traindata[:,:2]=qtypem
traindata[:,2:4]=numsdm
traindata[:,4:5]=tagm
traindata[:,5:6]=tagtm

#traindata[:,2:3]=sim


train=traindata[::2,:-1]
traint=targets[::2,:]
traint=np.squeeze(traint)

test = traindata[1::2,:-1]
testt = targets[1::2,:]
testt=np.squeeze(testt)




'''
random forest
'''
from sklearn.ensemble import RandomForestClassifier
# Create a random forest classifier. By convention, clf means 'classifier'
clf = RandomForestClassifier( warm_start =True, n_jobs=-1)
# Train the classifier to take the training features and learn how they relate
# to the training y (the species)
clf.fit(train, traint)


clf.score(test,testt)










'''
predict  23:09-23:16
'''

'''
read test data
'''
testdata=[]

#for i in range(12):  #0-11
testfilename="../final/owndata11"
predictedfilename="predict_11"  
with open(testfilename) as f:
    for line in f:
        line=line[:-3]
        line=unicode(line,errors='ignore')
        line=unicode.encode(line)
        line=repr(line).replace("\\"," ")
        line=line.replace('`',' ')
        line=line.replace("'"," ")
        line=line.replace('`s',' ')
        line=line.replace('(',' ')
        line=line.replace(')',' ')
        line=line.replace('?',' ')
        line=line.replace('#',' ')
        testdata.append(line.split(',"'))



'''
find question type of each question
'''
questionType=('why','what','where','who','whom',\
'whose','when','can','could','may','might','if','will','would','has','have',\
'do','doe','did','am','is','was','were',)

qt=dict()
for i in range(len(questionType)):
    qt[questionType[i]]=i

sqtype=[]                          #*******************************2
for i in range(len(testdata)):
    temp=[0]*2
    for t in questionType:
        if t in testdata[i][1]:
            temp[0]+=qt[t]
        if t in testdata[i][2]:
            temp[1]+=qt[t]
    sqtype.append(temp)


'''
# of same words 
# of diff words in two question, 
'''
import copy

testnumsd=[]               #************************************2
testcopy=[]
for d in testdata:
    temp=[]
    q1,q2=d[1].split(),d[2].split()
    ns=0
    l=copy.copy(q1)        
    for n in l: #find the same number
        if n in q1 and n in q2:
            ns+=1
            q1.remove(n)
            q2.remove(n)
    nd=((len(q1)-len(q2)))**2
    dif=[]
    dif.append(q1)
    dif.append(q2)
    testcopy.append(dif)    
    temp.append(ns)
    temp.append(nd)
    testnumsd.append(temp)

'''
delete not important words from the traincopy which is after delete the same words list
'''
important=['VB', 'NN',  'MD', 'JJ','W'] 


def delete_nonimportant(l, flag):
    tem=[]
    for w in pos_tag(l):
        for i in important:
            if i in w[1]:
                tem.append(w[flag])                
    return tem

testTag=[] 

for i in range(len(testcopy)):
    testcopy[i][0]=delete_nonimportant(testcopy[i][0],0)      #question 1, words
    testcopy[i][1]=delete_nonimportant(testcopy[i][1],0)      #question 2, words
    emp=[]
    emp.append(delete_nonimportant(testcopy[i][0],1) )     #question 1, tag
    emp.append(delete_nonimportant(testcopy[i][1],1) )     #question 2, tag
    testTag.append(emp)

'''
number of same tags
'''
testsamtags=[]  #***************************1

def find_same_tags(l1,l2):
    tem=0
    l=copy.copy(l1)
    for t in l:
        if t in l2:
            l1.remove(t)
            l2.remove(t)
            tem+=1
    return tem


for i in range(len(testTag)):
     testsamtags.append(find_same_tags(testTag[i][0],testTag[i][1]))

'''
 tag difference 
'''
testtagdiff=[]    #*********************************1

for i in range(len(testTag)):
    testtagdiff.append( (len(testTag[i][0])- len(testTag[i][1]))**2)


ids=[]

for d in testdata:
    ids.append(int(''.join(d[0].split())))

x=len(ids)

ids=np.reshape(ids, (x,1))

testmatrix=np.zeros((x,6))
sqtypem=np.reshape(sqtype, (x,2))
testnumsdm=np.reshape(testnumsd,(x,2))
testtagm=np.reshape(testsamtags, (x,1))
testtagtm=np.reshape(testtagdiff, (x,1))



testmatrix[:,:2]=sqtypem
testmatrix[:,2:4]=testnumsdm
testmatrix[:,4:5]=testtagm
testmatrix[:,5:]=testtagtm


# Apply the classifier we trained to the test data (which, remember, it has never seen before)

predicted=clf.predict_proba(testmatrix)


result=np.zeros((x,2))
result[:,:1]=ids
result[:,1:]=predicted[:,1:]

np.savetxt(predictedfilename, result, delimiter=',', fmt='%d,%1.2f')          #save predicted result to file.
print 'file:',predictedfilename,' is saved'



