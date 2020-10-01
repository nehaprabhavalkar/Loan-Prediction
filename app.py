from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
server = app.server

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
	dataset = pd.read_csv('train.csv')
	dataset['Gender'] = dataset['Gender'].fillna('Male')
	dataset['Married'] = dataset['Married'].fillna('Yes')
	dataset['LoanAmount'] = dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean())
	dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].fillna(360.0)
	dataset['Credit_History'] = dataset['Credit_History'].fillna(1.0)
	dataset['Self_Employed'] = dataset['Self_Employed'].fillna('No')
	dataset['Dependents'] = dataset['Dependents'].fillna(0)
	cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status','Loan_ID','Dependents']

	for i in cols:
		dataset[i] = pd.factorize(dataset[i])[0]

	dataset = dataset.drop(columns=['Education','Property_Area'])

	X = dataset.iloc[:,1:10]
	y = dataset.iloc[:,10]

	X_train , X_test , Y_train , Y_test = train_test_split(X,y,test_size=0.2,random_state=0)
	classifier = RandomForestClassifier(n_estimators=20)
	classifier.fit(X_train,Y_train)
	


	if request.method == 'POST':
		lid = request.form['lid']
		gender = request.form['gender']
		married = request.form['lid']
		dep = request.form['dep']
		edu = request.form['edu']
		selfemp = request.form['selfemp']
		appinc = request.form['appinc']
		coapp = request.form['coapp']
		loan = request.form['loan']
		loanterm = request.form['loanterm']
		credhist = request.form['credhist']
		proparea = request.form['proparea']

		if gender == 'Male':
			gender = 0
		else:
			gender = 1

		if married == 'Yes':
			married = 1
		else:
			married = 0

		if selfemp == 'Yes':
			selfemp = 1
		else:
			selfemp = 0

		df = pd.DataFrame(columns =['Gender','Married','Dependents','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History',]) 

		df.loc[0] = list = [gender, married,dep,selfemp,appinc,coapp,loan,loanterm,credhist] 
		
		prediction = classifier.predict(df)
		return render_template('output.html',prediction = prediction)

if __name__=='__main__':
	app.run(debug=True)
