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
-------------------------------------------------------------------------
## 2. Analyzing the sentiment of product reviews (Classification model)
-------------------------------------------------------------------------
* Process - Intelligent System : All reviews -> Break into sentences -> (sentiment classifier)look for particular words -> Output(+/-);
* Simple threshold classifier : Sentence from review(input x) -> feed with list of positive and negative word -> count the positive and negative words; If more + words -> positive review else a negative review;
* Linear classifier : It takes all the words and adds weights to them. Sentence from x review-> feed with list of words and weights and the score is computed; score = weight of word1 * occurence of word1 + weight of word2 * occurence of word2 + ...;
* Decision Boundary : A line of segregation between positive and negative reviews, decision boundary is 0;
* Classification Error = fraction of mistakes = # mistakes / total # of sentences = 1 - accuracy;
* Accuracy = fraction of correct predictions = # correct / total # of sentences = 1 - error;
* Confusion Matrix - The relation between true label and predicted label; (True Positive, False Negative, False Positive, True Negative);

#### Algorithm
------------------------------------
* Read the product review data. (name, review, rating)
* Create a word count vector for each review - tokenizing/separating the words. (name, review, rating, word_count)
* Extract the most popular product and explore it.
* Build a sentiment classifier. rating (4,5) -> positive; rating(1,2) -> negative; rating(3) -> removed (data engineering); (name, review, rating, word_count, sentiment)
  * Spliting the data - 80%/20% - training / test set;
  * target="sentiment"; feature="word_count"; algorithm="logistic_classifier", input=training_data, validation=test_data;
* Evaluate the sentiment classifier model w.r.t test data, metric = roc(confusion matrix);
* Predict the sentiment of the most popular product using the trained model.

-------------------------------
## 3. Retrieving Wikipedia articles (Clustering & Retrieval)
-------------------------------
* Similarity document retrieval : Most popular : "Bag of words" model.
* Bag of Words : Order of words is ignored, count the number of instances of words and create a vector. The word count vectors are taken from the various documents. The summation of element-wise multiplication is high for similar documents.
* TF-IDF - Term frequency inverse document frequency - it is the trade off between the local frequency and global rarity.
* TF - look locally - count the number of words within the document (word count vector);
* IDF - downweight the vector. All documents in the corpus a looked through - compute - log(#doc / (1 + #doc using the focus words))
* It's low for frequently occuring word and high for rarely occuring words.

#### Algorithms:
-------------------
###### Nearest neighbour model: Have a query article and a corpus to search articles from.
* Need to specify deistance metrics; Output -> collection of related articles.
###### 1-Nearest neighbour model : Input - Query articles; Output - Similar articles; 
    Search over the corpus  






































