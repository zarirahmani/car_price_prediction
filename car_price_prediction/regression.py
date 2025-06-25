
import pandas as pd
import numpy as np
import pickle


# Data Preparation

cars = pd.read_csv('cars.csv')

cars.columns = cars.columns.str.lower().str.replace(' ', '_')

strings = list(cars.dtypes[cars.dtypes == 'object'].index)
for col in strings:
    cars[col] = cars[col].str.lower().str.replace(' ', '_', regex=True)


# Spliting the data into train, validation and test sets

n_data = len(cars)
n_val = int(n_data * 0.2)  # 20% of the data for validation
n_test = int(n_data * 0.2)  # 20% of the data for testing
n_train = n_data - n_val - n_test  # Remaining data for training

np.random.seed(42)  # To make it reproducible
idx = np.arange(n_data)
np.random.shuffle(idx)

cars_train = cars.iloc[idx[:n_train]]
cars_val = cars.iloc[idx[n_train:n_train + n_val]]
cars_test = cars.iloc[idx[n_train + n_val:]]

cars_train = cars_train.reset_index(drop=True)
cars_val = cars_val.reset_index(drop=True)
cars_test = cars_test.reset_index(drop=True)

y_train = np.log1p(cars_train.msrp.values)   # Let's make sure we apply log transformation to the target variable.
y_val = np.log1p(cars_val.msrp.values)
y_test = np.log1p(cars_test.msrp.values)


# Let's drop the msrp column from the feature set.
del cars_train['msrp']
del cars_val['msrp']
del cars_test['msrp']


base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg','popularity']

categorical_variables = ['make', 'engine_fuel_type', 'transmission_type', 'driven_wheels', 'market_category', 'vehicle_size', 'vehicle_style']

categories = {}

for c in categorical_variables:
    categories[c] = list(cars[c].value_counts().head().index)


def train_linear_regression(X, y):
    ones = np.ones(X.shape[0]) # Create a column of ones for the intercept term.
    X = np.column_stack([ones, X])  # Stack the column of ones with the feature matrix.

    XTX = X.T.dot(X)  # Compute the Gram matrix.
    XTX_inv = np.linalg.inv(XTX)  # Compute the inverse of the Gram matrix.
    w_full = XTX_inv.dot(X.T).dot(y)  # Compute the weights using the normal equation.
    return w_full[0], w_full[1:]  # Return the intercept and the coefficients. 


X_train = cars_train[base].values  # Extract the feature values as a NumPy array

# I will check whether we have missing values. I will fill them with 0s.
X_train = cars_train[base].fillna(0).values

w0, w = train_linear_regression(X_train, y_train)

# Now, I can use weights to make predictions.
y_pred = w0 + X_train.dot(w)
y_pred

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


def rmse(y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)


X_train = prepare_X(cars_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(cars_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)


# Training the model

def train_linear_regression_reg(X, y, r = 0.001): # r is the regularization parameter.
    ones = np.ones(X.shape[0]) # Create a column of ones for the intercept term.
    X = np.column_stack([ones, X])  # Stack the column of ones with the feature matrix.

    XTX = X.T.dot(X)  # Compute the Gram matrix.
    XTX = XTX + r * np.eye(XTX.shape[0])  # Add regularization term to the Gram matrix.
    
    XTX_inv = np.linalg.inv(XTX)  # Compute the inverse of the Gram matrix.
    w_full = XTX_inv.dot(X.T).dot(y)  # Compute the weights using the normal equation.
    return w_full[0], w_full[1:]  # Return the intercept and the coefficients.


X_train = prepare_X(cars_train)
w0, w = train_linear_regression_reg(X_train, y_train, r = 0.001)

X_val = prepare_X(cars_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

r = 0.001
X_train = prepare_X(cars_train)
w0, w = train_linear_regression_reg(X_train, y_train, r = r)

X_val = prepare_X(cars_val)
y_pred = w0 + X_val.dot(w)
score = rmse(y_val, y_pred)
score

# I will combine the training and validation sets.

cars_full_train = pd.concat([cars_train, cars_val], ignore_index=True)

X_full_train = prepare_X(cars_full_train)
y_full_train = np.concatenate([y_train, y_val])
w0, w = train_linear_regression_reg(X_full_train, y_full_train, r = 0.001)

# I will prepare the test set.
X_test = prepare_X(cars_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)

with open('linear_regression_model.pkl', 'wb') as f_out:
    pickle.dump((w0, w, y_pred), f_out)
