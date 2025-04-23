import pickle 
from flask import Flask,request,jsonify,url_for,render_template

import numpy as np
import pandas as pd
app=Flask(__name__)
##load the model 
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html', prediction_text=None)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        input_array = np.array(list(data.values())).reshape(1, -1)
        print(f"Input shape: {input_array.shape}")
        new_data = scaler.transform(input_array)
        output = regmodel.predict(new_data)
        return jsonify({'prediction': output[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text=f"THE HOUSE PRICE PREDICTION IS: {output}")




                                                                                             



if __name__ == "__main__":
    app.run(debug=True)

    