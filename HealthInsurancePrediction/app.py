from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

mul_reg = open("HealthInsurancePrediction.pkl", "rb")
linear_model,sc = joblib.load(mul_reg)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    print("I was here ")
    if request.method == 'POST':
        print(request.form.get('Age'))
        try:
            Age = float(request.form['Age'])
            Sex = request.form['Sex']
            BMI = float(request.form['BMI'])
            Childrens = float(request.form['Childrens'])
            Smoker = request.form['Smoker']
            Region = request.form['Region']
            if Sex.lower() == 'male':
              Gender=1
            else:
              Gender=0
            if Smoker.lower() == 'yes':
              smo=1
            else:
              smo=0
            if Region.lower == 'northeast':
              pred_args = [1,0,0,0,Age, Gender, BMI, Childrens, smo]
            elif Region.lower == 'northwest':
              pred_args = [0,1,0,0,Age, Gender, BMI, Childrens, smo]
            elif Region.lower == 'southwest':
              pred_args = [0,0,0,1,Age, Gender, BMI, Childrens, smo]
            else:
              pred_args = [0,0,1,0,Age, Gender, BMI, Childrens, smo]
            
            print('Arguments values: ',pred_args)
            pred_args_arr = np.array(pred_args)
            pred_args_arr=sc.transform(pred_args_arr)
            print('Arguments values after performing Standardization: ',pred_args_arr)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            model_prediction = linear_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)
        except ValueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html', prediction = model_prediction)


if __name__ == "__main__":
    app.run(host='0.0.0.0')
