'''
IRIS DATASET
'''

# required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# read the dataset
data = pd.read_csv('data.csv')
print(data.head())

print('\n\nColumn Names\n\n')
print(data.columns)

#label encode the target variable
encode = LabelEncoder()
data.Species = encode.fit_transform(data.Species)

print(data.head())

# train-test-split   
train , test = train_test_split(data,test_size=0.2,random_state=0)

print('shape of training data : ',train.shape)
print('shape of testing data',test.shape)

# seperate the target and independent variable
train_x = train.drop(columns=['Species'],axis=1)
train_y = train['Species']

test_x = test.drop(columns=['Species'],axis=1)
test_y = test['Species']

# create the object of the model
model = LogisticRegression()

model.fit(train_x,train_y)

predict = model.predict(test_x)

print('Predicted Values on Test Data',encode.inverse_transform(predict))

print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(test_y,predict))

