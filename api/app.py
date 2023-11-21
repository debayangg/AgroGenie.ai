from flask import Flask, render_template, request
from ._myClass import LogisticRegression
import numpy as np
import os
from pickle import load

app = Flask(__name__,
            static_url_path='', 
            static_folder='../static',
            template_folder='../templates')

dict_values = load(open('api/imp.pkl', 'rb'))
loaded_model=LogisticRegression()
loaded_model.dict_crop=dict_values
path=os.getcwd()

#print(path)

@app.route('/')
def home():
    return render_template('hello.html')


@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(-1,1)

    prediction = loaded_model.prediction(single_pred)
    prediction=int(prediction)

    crop_dict = {1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas', 6: 'mothbeans',
                 7: 'mungbean', 8: 'blackgram', 9: 'lentil', 10: 'pomegranate', 11: 'banana', 12: 'mango',
                13: 'grapes', 14: 'watermelon', 15: 'muskmelon', 16: 'apple', 17: 'orange', 18: 'papaya',
                19: 'coconut', 20: 'cotton', 21: 'jute', 22: 'coffee'}

    if prediction in crop_dict:
        crop = crop_dict[prediction]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    
    return render_template('hello.html', prediction=result, img=crop+".png")

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
