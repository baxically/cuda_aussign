import numpy as np
import csv
from PIL import Image
from sklearn.decomposition import PCA
import random

X=[]
Y=[]
trainarray=[]
testarray=[]
label=[]
pixels=[]
temp=[]
temp2=[]
i=0
words = ['alive-', 'all-', 'answer-', 'boy-', 'building-', 'buy-', 'change_mind_-', 'cold-', 'come-', 'computer_PC_-', 'cost-', 'crazy-', 'danger-', 'deaf-', 'different-', 'draw-', 'drink-', 'eat-', 'exit-', 'flash-light-', 'forget-', 'girl-', 'give-', 'glove-', 'go-', 'God-', 'happy-', 'head-', 'hear-', 'hello-', 'his_hers-', 'hot-', 'how-', 'hurry-', 'hurt-', 'I-', 'innocent-', 'is_true_-', 'joke-', 'juice-', 'know-', 'later-', 'lose-', 'love-', 'make-', 'man-', 'maybe-', 'mine-', 'money-', 'more-', 'name-', 'no-', 'Norway-', 'not-my-problem-', 'paper-', 'pen-', 'please-', 'polite-', 'question-', 'read-', 'ready-', 'research-', 'responsible-', 'right-', 'sad-', 'same-', 'science-', 'share-', 'shop-', 'soon-', 'sorry-', 'spend-', 'stubborn-', 'surprise-', 'take-', 'temper-', 'thank-', 'think-', 'tray-', 'us-', 'voluntary-', 'wait_notyet_-', 'what-', 'when-', 'where-', 'which-', 'who-', 'why-', 'wild-', 'will-', 'write-', 'wrong-', 'yes-', 'you-', 'zero-']
# print(len(words))
while i<9:
    folder_name='tctodd{}'.format(i+1)
    for index in range(95):
        title=words[index]
        j=1
        while j<4:
            filename=('./{}/{}{}.tsd'.format(folder_name,title,j))
            # print(filename)
            Y.append(index)
            with open(filename) as file:
                rd = csv.reader(file, delimiter="\t")
                for row in rd:
                    X.append(np.array(row))      
            j=j+1
    i=i+1
X_train = np.array(X)
# Y_train = np.array(Y)
# X_train = X_train.reshape(-1,22)
print(X_train.shape)
print(len(Y))
traindata = "aussign_train.csv"
testdata = "aussign_test.csv"

nShuffle=list(range(0,2565))
np.random.shuffle(nShuffle)
numtrainsample=1796
i=0;
for x in nShuffle[0:numtrainsample]:
    label.append(Y[x])
    temp = np.concatenate((label, X_train[x].flatten()))
    trainarray.append(temp)
    label.pop()
    
    
    
for y in nShuffle[numtrainsample:2565]:
    label.append(Y[y])
    temp2 = np.concatenate((label, X_train[y].flatten()))
    testarray.append(temp2)
    label.pop()

 
with open(traindata, mode = 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerows(trainarray)
    
with open(testdata, mode = 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csvwriter.writerows(testarray)
        

