# importing required libraries
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from keras.utils import np_utils

# Loading dataset
data = np.loadtxt("data.txt")

# Splitting dataset
X_train = data[:4646,:12]
Y_train = data[:4646,12:13]
X_test = data[4646:6936,:12]
Y_test = data[4646:6936,12:13]
len_y_train = len(Y_train)
len_y_test = len(Y_test)

print('Shape of training data :',X_train.shape)
print('Shape of testing data :',Y_test.shape)


# Data Preprocessing
for i in range(0,len_y_train):
    if(Y_train[i]==1000):
        Y_train[i] = 3
        Y_train[i] = int(Y_train[i])
    elif(Y_train[i]==100):
        Y_train[i] = 2
        Y_train[i] = int(Y_train[i])
    elif(Y_train[i]==10):
        Y_train[i] = 1
        Y_train[i] = int(Y_train[i])
    elif(Y_train[i]==1):
        Y_train[i] = 0
        Y_train[i] = int(Y_train[i])

for i in range(0,len_y_test):
    if(Y_test[i]==1000):
        Y_test[i] = 3
        Y_test[i] = int(Y_test[i])
    elif(Y_test[i]==100):
        Y_test[i] = 2
        Y_test[i] = int(Y_test[i])
    elif(Y_test[i]==10):
        Y_test[i] = 1
        Y_test[i] = int(Y_test[i])
    elif(Y_test[i]==1):
        Y_test[i] = 0
        Y_test[i] = int(Y_test[i])

def res(x):
    if(x==0):
        return 'ThunderStorm'
    elif(x==1):
        return 'Rainy'
    elif(x==2):
        return 'Foggy'
    else:
        return 'Sunny'

model = GaussianNB()


# fit the model with the training data
model.fit(X_train,Y_train.ravel())

# predict the target on the train dataset
predict_train = model.predict(X_train)
print('Target on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(Y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train*100)

# predict the target on the test dataset
predict_test = model.predict(X_test)

# print('Target on test data',[predict_test for predict_test in predict_test]) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(Y_test,predict_test)
print('accuracy_score on test dataset : ', accuracy_test*100)

# Load Prediction Data
pre_data = np.loadtxt('predict.txt').reshape(1, -1)
pre_data=pre_data.astype('int32')




# Print Input & Prediction
print('Input: ',pre_data)
print ("\n \t \t \t Weather would be",res(model.predict(pre_data)))


# This is Naive bayes network

