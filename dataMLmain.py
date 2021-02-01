#MachineLearning algorithm to detect if someone has Diabetes

#Libraries
import streamlit as sl 
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

#Title
sl.write("""
# Detection of Diabetes

Using Machine Learning and Python, this will help detect if someone has Diabetes

	""")

#Get the data
df = pd.read_csv('C:/Users/Diabetes.csv')

#Show Data file informtion
sl.subheader('Data Info')
sl.dataframe(df)

#Show Statistics
sl.subheader('Data Statistics')
sl.write(df.describe())

#Show Chart of Data
sl.subheader('Data Chart')
chart = sl.bar_chart(df)

#Split the data into ind 'x' and dep 'y' variables
X=df.iloc[:, 0:8].values
Y=df.iloc[:, -1].values

#Split data into 70% Training and 30 % Testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

#Get the input variables from the user
def get_input():
	pregancies = sl.sidebar.slider('pregancies', 0, 15, 1)
	glucose = sl.sidebar.slider('glucose', 0, 200, 110)
	blood_pressure = sl.sidebar.slider('blood_pressure', 0, 124, 75)
	skin_thickness = sl.sidebar.slider('skin_thickness', 0, 99, 20)
	insulin = sl.sidebar.slider('insulin', 0.0, 850.0, 35.0)
	BMI = sl.sidebar.slider('BMI', 0.0, 67.2, 33.0)
	DFF = sl.sidebar.slider('DFF', 0.076, 2.43, 0.345)
	age = sl.sidebar.slider('age', 21, 81, 29)


	#Store an input into a variable
	user_inputData = {'pregancies': pregancies,
		'glucose': glucose,
		'blood_pressure': blood_pressure,
		'skin_thickness': skin_thickness, 
		'insulin': insulin,
		'BMI': BMI,
		'DFF': DFF,
		'age': age
	}

	#Data -> Data Frame
	features = pd.DataFrame(user_inputData, index = [0])
	return features

#Store user data
user_input = get_input()

#Set a subheader and display inputted values
sl.subheader('User Data Input')
sl.write(user_input)

#Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show model Accuracy
sl.subheader('Model Test Accuracy Score')
sl.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

#Store the models predictions 
predict = RandomForestClassifier.predict(user_input)

#Set SubHeader and display classification
sl.subheader('Diabetes Detection: 0-> No Diabetes | 1->Has Diabetes')
sl.write(predict)
