#I want to write a function to create a web servie and I use Flask for that.

import pickle
import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify

with open("linear_regression_model.pkl", "rb") as f_in:
    w0, w, categories = pickle.load(f_in)

base = ["engine_hp", "engine_cylinders", "highway_mpg", "city_mpg","popularity"]

def prepare_X(cars):
    cars = pd.DataFrame([cars])  # Ensure cars is a DataFrame
    cars = cars.copy()  # Create a copy to avoid modifying the original DataFrame
    features = base.copy()  # I don"t want to modify the base list

    cars["age"] = 2017 - cars["year"]
    features.append("age")

    for v in [2, 3, 4]:
        cars["num_doors_%s" % v] = (cars["number_of_doors"] == v).astype(int)
        features.append("num_doors_%s" % v)

    for c, values in categories.items():
        for v in values:
            cars["%s_%s" % (c, v)] = (cars[c] == v).astype(int)
            features.append("%s_%s" % (c, v))
    
    cars_num = cars[features]
    cars_num = cars_num.fillna(0) 
    X = cars_num.values
    return X

app = Flask("price")
@app.route("/predict", methods=["POST"])  # I want to access this function using GET method and the endpoint will be /ping.
def predict():
    request.get_json()  # Get the JSON data from the request
    car = request.json  # Get the JSON data from the request
    X = prepare_X(car)   # For the ideal project, I will put it in a separate function and also the other two statements.
    y_pred = w0 + X.dot(w)
    price = np.expm1(y_pred[0])

    price = {
        'predicted_price': price}
    return jsonify(price)

if __name__ == "__main__": 
    app.run(debug=True, host="0.0.0.0", port=9696)  # This will run the Flask app on all available IP addresses on port 9696.


# curl is a command-line tool to communicate with web services.

