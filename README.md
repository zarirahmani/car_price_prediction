# 🔭 Machine Learning Course


## 🛻 Car Price Prediction Model

In this project, I have designed a regression model to predict the price of used cars based on features such as age of the car, mileage, make, engine horsepower, et. which included both numerical and categorical variables. The goal is to build a regression model that generalises well on unseen data. The model was trained and validated on a dataset of used car listings and optimised using regularisation and hyperparameter tuning techniques. The data was obtained from [the Kaggle competition](https://www.kaggle.com/datasets/CooperUnion/cardataset). The data is also provided in this repository under car_price_prediction folder. The Python notebook is named "regression".

### 🪜 Key steps included:


- **Preprocessing and Data Cleaning**
- **Training the model**
- **Evaluating the model with RMSE**
- **Feature Engineering**
- **Regularisation**
- **Model Tuning**

### 🧪 Validation Method

I split the data into training, validation and test sets. training set was used to train the model, validation set for evaluation and test set for the final evaluation. 

### 🔬 Evaluation Metric

**Root Mean Squared Error (RMSE)**  was used to evaluate the model which is a common tool for evaluating regression models. The lower RMSE, the better performance of the model.

### 🕵🏻‍♀️ Feature Engineering
To train the model, I first used the features that were numerical variables. To improve the performance of the model, I then added other features that were categorical. I transformed categorical variables into numerical variables (one-hot encoding) because only numerical variables can be used in a regression model. 

### 👩🏻‍🔧 Regularisation
To prevent overfitting, I used a regularisation technique called Ridge regression where I added a small value to the diagonal elements. To find this value I used the validation set.

### 💣 Model Tuning
To find the best regularisation hyperparameter value, I used validation set and validation RMSE and the final model was selected based on the best validation RMSE and generalization on the test set. 
