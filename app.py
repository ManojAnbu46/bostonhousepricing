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

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]

    # --- New Code to convert to INR ---
    prediction_in_usd = output * 1000  # Convert to dollars
    conversion_rate = 83  # 1 USD ≈ 83 INR
    prediction_in_inr = prediction_in_usd * conversion_rate

    # Convert to lakh or crore
    if prediction_in_inr >= 10000000:
        prediction_in_crore = prediction_in_inr / 10000000
        formatted_prediction = f"₹{prediction_in_crore:.2f} Crore"
    else:
        prediction_in_lakh = prediction_in_inr / 100000
        formatted_prediction = f"₹{prediction_in_lakh:.2f} Lakh"
    # --- End of New Code ---

    return render_template("home.html", prediction_text=f"THE HOUSE PRICE PREDICTION IS: {formatted_prediction}")




                                                                                             



if __name__ == "__main__":
    app.run(debug=True)

    