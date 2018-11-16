
# coding: utf-8

# In[100]:

from numpy import genfromtxt
import pandas as pd
import numpy 
import scipy
import matplotlib.pyplot as plt

def ranking_function(T):
    
    temp_var = []
    w=numpy.zeros([T+1,763])
    w[0] = w_initial
    for t in range(T):
        w[t+1] = numpy.matmul(w[t],Normalized_M)
        w[t+1] = w[t+1]/numpy.sum(w[t+1])
    ind = w[t+1].argsort()[-25:][::-1]
    
    for i in range(0,len(ind)):
        temp_var.append({'team': TeamNames[ind[i]], 'w_final': w[t+1][ind[i]]})

    temp_var = pd.DataFrame(temp_var)
    return temp_var

team_scores = genfromtxt('CFB2017_scores.csv', delimiter=',')
team_scores = team_scores.astype(int)
TeamNames=[]
w_initial = numpy.random.uniform(0,1,[1,763])
M=numpy.zeros([763,763])

file = open('TeamNames.txt','r')
for i,line in enumerate(file):
    TeamNames.append(line.strip())
TeamNames=numpy.array(TeamNames)

for i in range(team_scores.shape[0]):

    x1 = team_scores[i][0]
    x2 = team_scores[i][1]
    x3 = team_scores[i][2]
    x4 = team_scores[i][3]
    
    if team_scores[i][1]>team_scores[i][3]:
        
        M[x1-1][x1-1] += 1 + (x2/(x2+x4))
        M[x3-1][x1-1] += 1 + (x2/(x2+x4))
        M[x3-1][x3-1] += 0 + (x4/(x2+x4))
        M[x1-1][x3-1] += 0 + (x4/(x2+x4))
    
    elif team_scores[i][1]<team_scores[i][3]: 
        
        M[x1-1][x1-1] += 0 + (x2/(x2+x4))
        M[x3-1][x1-1] += 0 + (x2/(x2+x4))
        M[x3-1][x3-1] += 1 + (x4/(x2+x4))
        M[x1-1][x3-1] += 1 + (x4/(x2+x4))
                
    else:
        
        M[x3-1][x1-1] +=  1 + (x2/(x2+x4))
        M[x3-1][x3-1] +=  1 + (x4/(x2+x4))
        M[x1-1][x1-1] +=  1 + (x2/(x2+x4))
        M[x1-1][x3-1] +=  1 + (x4/(x2+x4))

row_wise_sum = M.sum(axis=1)
Normalized_M =  M / row_wise_sum[:, numpy.newaxis]

print("Top 25 teams for t=10")
result1 = ranking_function(10)
print(result1)

print("Top 25 teams for t=100")
result1 = ranking_function(100)
print(result1)

print("Top 25 teams for t=1000")
result1 = ranking_function(1000)
print(result1)

print("Top 25 teams for t=10000")
result1 = ranking_function(10000)
print(result1)


# In[107]:

T = 10000
a,b = scipy.sparse.linalg.eigs(Normalized_M.T,k=1,sigma=1.0)
b = b/numpy.sum(b)
w = numpy.zeros([T+1,763])
w[0] = w_initial
for t in range(T):
    w[t+1] = numpy.matmul(w[t],Normalized_M)
    w[t+1] = w[t+1]/(numpy.sum(w[t+1]))

answer = []
for i in range(1,T):
    temp_var = numpy.linalg.norm(w[i].reshape([763,1])-b,ord=1)
    answer.append(temp_var)

plt.figure(figsize=(8,8))    
plt.plot(numpy.arange(1, T),answer)
plt.xlabel("iterations")
plt.title(" |w(t) - w(inf)| as a function of t")
plt.show()


# In[ ]:


import numpy
def nmf_function(data_file, vocab_file, T):
    
    vocab_length = 3012
    doc_length = 8447
    d = 25
    W = numpy.random.uniform(1,2,(vocab_length,d))
    H = numpy.random.uniform(1,2,(d,doc_length))
    X = numpy.zeros((vocab_length,doc_length))
    
    temp_var = 0
    with open(data_file) as f:
        for row in f:
            nwords = row.rstrip('\n').split(',')
            for y in nwords:
                a,b = [int(x) for x in y.split(':')]
                X[a-1,temp_var] = b
            temp_var+=1
    
    with open(vocab_file) as file:
        words = numpy.array([a.rstrip('\n') for a in file.readlines()])


    obj_list = []
    WH = W.dot(H)
    temp_matrix = X/(WH+1e-16)

    for i in range(T):

        var_temp = numpy.sum(W,axis=0).reshape(d,1)
        H = numpy.multiply(H, W.T.dot(temp_matrix))/ var_temp
        WH = W.dot(H)
        temp_matrix = X/(WH+1e-16)
        var_temp = numpy.sum(H,axis=1).reshape(1,d)
        W = numpy.multiply(W, temp_matrix.dot(H.T))/ var_temp

        WH = W.dot(H)
        temp_matrix = X/(WH+1e-16)

        obj = numpy.sum(numpy.multiply(numpy.log(1/(WH+1e-16)),X) + WH)
        obj_list.append(obj)

        W = W / (numpy.sum(W, axis=0).reshape(1,-1))
        word_index = numpy.apply_along_axis(lambda x: numpy.argsort(x)[-10:][::-1],axis=0,arr=W)
        answer = pd.DataFrame(index=range(10),columns=['topic %d'%i for i in range(1,26)])
        for i in range(25):
            answer.iloc[:,i] = list(zip([format(x, '.3f') for x in W[word_index[:,i],i]],words[word_index[:,i]]))
        
        answer.to_csv('q2_b.csv',index=False)
        
    plt.figure()
    plt.plot(range(1,T+1),obj_list)
    plt.xticks(numpy.linspace(1,T,5))
    plt.xlabel('Iterations')
    plt.title('Objective function for iterations')
    plt.show()
    





# In[ ]:



