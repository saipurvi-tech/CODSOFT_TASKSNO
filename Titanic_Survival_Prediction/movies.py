import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#importing the data from csv to pandas data file 
titanic_data = pd.read_csv("Titanic-Dataset.csv")

#now we are going to perform EDA to understand out dataset
#print(titanic_data.head())
#print(titanic_data.columns)
#print(titanic_data.shape)
#print(titanic_data.info())  #we get to know that there are few missing values, so let us check the number of missing columns
#print(titanic_data.isnull().sum())   #we have majority missing values in Cabin, age and Embarkmed

titanic_data = titanic_data.drop(columns='Cabin') #dropping the cabin column as it has majority missing values

#replacing the missing values in the age column with the mean value of the age column
titanic_data['Age'] = titanic_data['Age'].fillna(titanic_data['Age'].mean())
#print(titanic_data.isnull().sum())
#replacing the missing values in the Embarked column with the mode value of the Embarked column
titanic_data['Embarked'] = titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])
#print(titanic_data.isnull().sum())

#analysing the statistical data of the dataset
#print(titanic_data.describe())

#now let us find the number of people who have survived and those who have not 
#print(titanic_data['Survived'].value_counts())
#out of 891 people, 549 people have not survived and 342 people have survived

#now let us perform data visualization to understand our dataset better
sns.set()
#sns.countplot(x='Survived', data=titanic_data)
#plt.show()

#sns.countplot(x='Sex', data=titanic_data)
#plt.show()

#checking number of survivers based on gender
#sns.countplot(x='Sex', hue='Survived', data=titanic_data)
#plt.show()

#sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
#plt.show()

#because we cannot feed string values to pur model, we are indicating male as 0 and female as 1
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
#even the Embarked column has string values, so we are indicating S as 0, C as 1 and Q as 2
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S':0, 'C':1, 'Q':2})
#print(titanic_data.head())

#seperating the features and target 
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
Y = titanic_data['Survived'] 
#print(X.head())
#print(Y.head())

#splitting the data into training and testing sets to train our model 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
#print(X.shape, X_train.shape, X_test.shape)

#now we are model training, using the logistic regression model to train
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, Y_train)

#checking the accuracy score of our model on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
#print('Accuracy score of the training data : ', training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
#print('Accuracy score of the test data : ', test_data_accuracy)

#now let us ask for the user input to predict whether the person will survive or not
def predict_survival(model, feature_names):
    pclass = int(input("\nEnter the passenger class (1=1st, 2=2nd, or 3=3rd class): "))
    sex = int(input("Enter the gender (0=male, 1=female): "))
    age = float(input("Enter the age of the passenger: "))
    sibsp = int(input("Enter the number of siblings/Spouses Abord: "))
    parch = int(input("Enter the number of parents/children Abord: "))
    fare = float(input("Enter ticket fare: "))
    embarked = int(input("Enter port of embarkation (S=0, C=1, Q=2): "))

    passenger_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],columns=feature_names)
    prediction = model.predict(passenger_data)[0]
    print("\nTHE PREDICTION RESTULT:\n")
    if prediction == 1:
        print("State: Survived\n\n")
    else:
        print("Status: Not Survived\n\n")

predict_survival(model, X.columns) 