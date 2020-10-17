from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('admission1.pkl','rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods = ['Post'])
def predict():
    if request.method == "POST":
        n1 = request.form['GRE']
        n2 = request.form['TOEFL']
        n3 = request.form['University_Rating']
        n4 = request.form['SOP']
        n5 = request.form['LOR']
        n6 = request.form['CGPA']
        n7 = request.form['Research']



        arr = [n1,n2,n3,n4,n5,n6,n7]
        data = np.array([arr])


        prediction = model.predict(data)[0]
    return render_template('index.html',result = (prediction*100).round(2))




if __name__ == '__main__':
	app.run(debug=True)