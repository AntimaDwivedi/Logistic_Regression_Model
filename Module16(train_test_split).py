# 1)-Evaluate using a Train and Test Set split

import warnings
warnings.filterwarnings(action="ignore")

import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

filename = 'accident_prone.data.csv'
names = ['ear','yawn','pulse','output']

dataframe = pd.read_csv(filename, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

test_data_size = 0.33
seed =7

X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, test_size=test_data_size,
                    random_state=seed  )# random_state=seed fixed the accuracy

model = LogisticRegression()
model.fit(X_train, Y_train)#fit trains the algorithm.
result = model.score(X_test, Y_test)#score algirithm ko test karayega

print(  "Accuracy= %f %%" % (result * 100)  )
#Accuracy is not fixed because it takes random 33% data.
