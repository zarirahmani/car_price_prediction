import pickle
import numpy as np

with open('linear_regression_model.pkl', 'rb') as f_in:
    w0, w, y_pred = pickle.load(f_in)

car = {
    'engine_hp': 200,
    'engine_cylinders': 4,
    'highway_mpg': 30,
    'city_mpg': 22,
    'popularity': 1000,
    'year': 2015,
    'number_of_doors': 4,
    'make': 'ford',
    'engine_fuel_type': 'regular_unleaded',
    'transmission_type': 'automatic',
    'driven_wheels': 'front_wheel_drive',
    'market_category': 'crossover',
    'vehicle_size': 'midsize',
    'vehicle_style': 'wagon'
}


price = np.expm1(y_pred[0])
print(f"Predicted price: ${price:.2f}")