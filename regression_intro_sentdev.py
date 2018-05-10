###Source: PythonProgramming.net
###Links: https://pythonprogramming.net/regression-introduction-machine-learning-tutorial/?completed=/machine-learning-tutorial-python-introduction/
###testing edit

import pandas as pd
import quandl, math
import numpy as np
##numpy will let array
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key ='6uqEjwn6Y2Z-zxvkxyC8'

df = quandl.get('WIKI/GOOGL') #get this from quandl.com

##print(df.head())

##grab features
##df is data frame
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

df['HL_PCT'] = (df['Adj. High']-df['Adj. Close']) / df['Adj. Close'] *100.0
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open']) / df['Adj. Open'] *100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']] # this is making df as variable

##print(df.head())

forecast_col = 'Adj. Close' #assigned Adj. Close as forecast_col variable
df.fillna(-99999, inplace =True) #dealing with N/A data

forecast_out = int(math.ceil(0.001*len(df)))
##print(len(df))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
##print(df.head())
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1)) #drop mean exclude , in this case 'label'
X = preprocessing.scale(X) #preprocess the all the value training and testing data
#X = X[:-forecast_out+1] #redefine X so that we only work with data same like matlab X=1:length-1
y = np.array(df['label']) #define y
##print(len(X),len(y))


############## Linear Regression Classifier ##############
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2) #creating cross_validation
clf = LinearRegression()
clf.fit(X_train, y_train) #training classifier
accuracy = clf.score(X_test, y_test) #testing classifier

print(accuracy)

############ SVM regression , SVR ##########
'''
clf2 = svm.SVR()
clf2.fit(X_train, y_train) #training classifier
accuracy2 = clf2.score(X_test, y_test) #testing classifier

print(accuracy2)
'''


             

             
             
         
