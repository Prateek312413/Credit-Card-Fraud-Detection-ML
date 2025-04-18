import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('Credit_card.csv')

# first 5 rows of the dataset
# print(credit_card_data.head())

# print(credit_card_data.tail())

# dataset informations
#print(credit_card_data.info())

# checking the number of missing values in each column
#print(credit_card_data.isnull().sum())

# distribution of legit transactions & fraudulent transactions
#print(credit_card_data['Class'].value_counts())

# This Dataset is highly unblanced

# 0 --> Normal Transaction

# 1 --> fraudulent transaction

# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# print(legit.shape)
# print(fraud.shape)

# statistical measures of the data
#print(legit.Amount.describe())

#print(fraud.Amount.describe())

# compare the values for both transactions
#print(credit_card_data.groupby('Class').mean())

# Under-Sampling

# Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions

# Number of Fraudulent Transactions --> 492


legit_sample = legit.sample(n=492)

new_dataset = pd.concat([legit_sample, fraud], axis=0)

# print(new_dataset.head())

# print(new_dataset.tail())

# print(new_dataset['Class'].value_counts())

# print(new_dataset.groupby('Class').mean())

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# print(X)

# print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# print(print(X.shape, X_train.shape, X_test.shape))

model = LogisticRegression()

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data : ', test_data_accuracy)

input_data = (43704,1.21094786957208,0.124804705710462,0.22583663864635,0.667968268965355,-0.407588675360832,-0.662221012901153,-0.15994737299361,0.0763867898704389,0.406295569947104,-0.253647734178919,-0.270197625736527,-1.00594917897938,-2.28721745934207,0.237079606571712,1.61947531291065,0.303095361566387,0.415542989476658,-0.442129334489197,-0.539965950329586,-0.271865562799685,-0.288538276662415,-0.919852701470252,0.188613744593241,-0.0343355639247329,0.058847802457702,0.132454304888077,-0.0229262094118139,0.0220891334935887,1.98)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
# print(prediction)

if (prediction[0]== 0):
  print('Normal Transaction')
else:
  print('Fradaulent Transaction')