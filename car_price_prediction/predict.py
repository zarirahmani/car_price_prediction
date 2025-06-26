import pickle
import numpy as np
import pandas as pd

with open('linear_regression_model.pkl', 'rb') as f_in:
    w0, w, categories = pickle.load(f_in)

base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg','popularity']

def prepare_X(cars):
    cars = cars.copy()  # Create a copy to avoid modifying the original DataFrame
    features = base.copy()  # I don't want to modify the base list

    cars['age'] = 2017 - cars.year
    features.append('age')

    for v in [2, 3, 4]:
        cars['num_doors_%s' % v] = (cars.number_of_doors == v).astype(int)
        features.append('num_doors_%s' % v)

    for c, values in categories.items():
        for v in values:
            cars['%s_%s' % (c, v)] = (cars[c] == v).astype(int)
            features.append('%s_%s' % (c, v))
    
    cars_num = cars[features]
    cars_num = cars_num.fillna(0) 
    X = cars_num.values
    return X

car = {
    'engine_hp': 300,
    'engine_cylinders': 4,
    'highway_mpg': 30,
    'city_mpg': 22,
    'popularity': 2300,
    'year': 2017,
    'number_of_doors': 2,
    'make': 'ford',
    'engine_fuel_type': 'regular_unleaded',
    'transmission_type': 'automatic',
    'driven_wheels': 'front_wheel_drive',
    'market_category': 'crossover',
    'vehicle_size': 'midsize',
    'vehicle_style': 'wagon'
}

X = prepare_X(car)
y_pred = w0 + X.dot(w)
price = np.expm1(y_pred[0])
print(f"Predicted price: ${price:.2f}")