
import requests

url = 'http://localhost:9696/predict'

car = {
    "engine_hp": 300,
    "engine_cylinders": 4,
    "highway_mpg": 30,
    "city_mpg": 22,
    "popularity": 2300,
    "year": 2017,
    "number_of_doors": 2,
    "make": "ford",
    "engine_fuel_type": "regular_unleaded",
    "transmission_type": "automatic",
    "driven_wheels": "front_wheel_drive",
    "market_category": "crossover",
    "vehicle_size": "midsize",
    "vehicle_style": "wagon"
}

price = requests.post(url, json = car)
print(price.json())
# A response [200] means the Flask prediction service received the POST request and responded successfully.

# I can only use WSGI server, so I will use gunicorn.
# And I use this command `gunicorn --bind 0.0.0.0:9696 predict:app`
# gunicorn doesn't work on Windows because it uses Linux specific features. Instead, I will use waitress: `pip install waitress`





