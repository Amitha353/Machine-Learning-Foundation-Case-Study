# Machine-Learning-Specialization

## I. Machine Learning Foundations : A Case Study Approach
* Predicting house prices
* Analyzing the sentiment of product reviews
* Retrieving Wikipedia articles
* Recommending songs
* Classifying images with deep learning

## 1. Predicting house prices (Regression Model to Predict House Prices)
-------------------------------------------------------------------------
* X -> feature, covariant, predictor, independent;
* Y -> observation, response, dependent;

#### Linear Regression Model (Fit a line through the model)
* fw(x) = w0 + w1*x; (w0 -> Intercept; w1 -> slope; parametrized function w = (w0,w1))
* Various lines are fit into the dataset and the line with mininum RSS cost is choosen.
* RSS(Residual Sum of Error) : The line is fit throught the dataset, and check how far the observation is from what the model predicted (fitted model).

#### Adding Higher order terms
--------------------------------
* Quadratic function fw(x) = w0 + w1*x + w2*x^2, 13th order polynomial can be a better fit for the dataset. Still a linear regression.

#### Algorithm
---------------
* Load the house sales data (condition, grade, sqft_above, sqft_basement,yr_built, yr_renovated, zipcode, lat, long, sqft_living, sqft_lot)
---------
* Explore the data: Create simple regression model (training/test data split - 80%/20%)
* Build the regression model: feature: "sqft_living"; target: price;
* Evaluate the simple model w.r.t test_data -> mean of the test price and evaluate -> max_error and rmse(root-mean-square error);
* Plot the Predictions and coefficient (sqft_living -> avg -> $282/sqft)
--------
* Explore the data with more features - bedrooms, bathrooms, sqft_living, sqft_lot, floors, zipcode; target - price;
* Evaluate the many_features model with test-data;
* Predict the price of the house;

