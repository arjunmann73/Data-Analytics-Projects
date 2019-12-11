'''
LOAN DATASET
'''

# required libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# read the dataset
data = pd.read_csv('loan.csv')
print(data.head())

print('\n\nColumn Names\n\n')
print(data.columns)

#label encode the target variable
encode = LabelEncoder()
data.Loan_Status = encode.fit_transform(data.Loan_Status)

# drop the null values
data.dropna(how='any',inplace=True)


# train-test-split   
train , test = train_test_split(data,test_size=0.2,random_state=0)



# seperate the target and independent variable
train_x = train.drop(columns=['Loan_ID','Loan_Status'],axis=1)
train_y = train['Loan_Status']

test_x = test.drop(columns=['Loan_ID','Loan_Status'],axis=1)
test_y = test['Loan_Status']

# encode the data
train_x = pd.get_dummies(train_x)
test_x  = pd.get_dummies(test_x)

print('shape of training data : ',train_x.shape)
print('shape of testing data : ',test_x.shape)

# create the object of the model
model = LogisticRegression()

model.fit(train_x,train_y)

predict = model.predict(test_x)

print('Predicted Values on Test Data',predict)

print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(test_y,predict))
