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
--------------------
* Read the product review data. (name, review, rating)
* Create a word count vector for each review - tokenizing/separating the words. (name, review, rating, word_count)
* Extract the most popular product and explore it.
* Build a sentiment classifier. rating (4,5) -> positive; rating(1,2) -> negative; rating(3) -> removed (data engineering); (name, review, rating, word_count, sentiment)
  * Spliting the data - 80%/20% - training / test set;
  * target="sentiment"; feature="word_count"; algorithm="logistic_classifier", input=training_data, validation=test_data;
* Evaluate the sentiment classifier model w.r.t test data, metric = roc(confusion matrix);
* Predict the sentiment of the most popular product using the trained model.

-------------------------------
## 3. Retrieving Wikipedia articles (Clustering & Retrieval) (Unsupervised Approach)
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
* Search over each article in the corpus  
* compute s = similarity(query article, corpus article)
* If s> Best_s, record doc_article = corpus article and ser Best_S=s; Return;
###### K-Nearest neighbors:
* Input - Query article; Output - List of K similar articles; (same as above);
###### K-means:
* Assume - Similarity : Distance to the cluster centers (smaller the distance - better)
* 1. Need to choose many clusters (k). Initialize the cluster centers.
* 2. Assign observations to the closest clusters - using Voronoi tessellation;
* 3. Revise the cluster centers as mean of assigned observations. (Initially the cluster centers are randomly initialized, therefore iterate on the observations inorder to retrieve a better cluster center that fits the data);
* 4. Repeat 2 and 3

#### Document Retrieval
* Load the text data (wikipedia);
* Explore the dataset - URI, name, text;
* Get word count for the focus article, sort and create a dictionary. - URI, name, text, word_count.
* compute tf_idf on the entire corpus. - URI, name, text, word_count, tfidf.
* Examine tf-idf for the focus article. sort w.r.t tfidf;
* compute the distance between articles to compare the similarity. (Lower the cosine distance, better the similarity)
* Build the nearest neighbour model - knn-model (Input -> people dataset; feature -> tfidf; label-> name;) Output -> clusters (similarity clusters);

-------------------------------
## 4. Recommender System (Recommending songs)
-------------------------------
* Personalization is transforming our experience in the world, connects users to items.
* Recommendations combine glbal and session interest, and recommendations must adapt to changing times and needs.
* Building a Recommender System:
###### Solution 1 : Popularity
* New articles - most read, most email, etc;
* Limitation - No personalization, results are based on the entire set of users/readers.
###### Solution 2 : Classification Model
* The model will be used to evaluate whether a user likes or dislikes a product.
                                                                                -------> Yes!
                                                                               |
* Input (User info, Purchase history, Product Info, Other info) -> Classifier -
                                                                               |
                                                                                -------> No
* Pros : Personalized(considers user info and purchase history); Features can capture context(time of day, etc);
* Limitation : Model dependent features may not be available. Not optimum model;
###### Solution 3 : Collaborative Filtering 
* Co-occurence Matrix - (People who bought this also bought that) - It stores users and items they bought. (#item X #item) matrix;
* Co-occurence matrix - built by iterating/searching through all the user history of purchases that have been made and count incrementing with each new product.

###### Making Recommendations using co-occurence matrix:
* Look as the focus product row in the matrix and extract it.
* Recommendations are made by sorting the listed vector and recommend the items with the largest count.
* Limitation: In case of popular items, they seem to domainate the recommendations.
* Therefore need to normalize co-occurence matrix;

###### Normalize co-occurences : Similarity Matrix
* Jaccard Similarity - normalize by popularity -> (# purchased by i and j) / (# purchased by i or j);
* Limitation : Only current page matters, no history recorded.

###### Weighted/Average of purchased items:
* A weighted average is computed based on having purchase history. User specific score for each item is computed for each item j in the inventory by combining similarities.
* Example : User has purchased - phone and phonecover, given this the probability of a user purchasing a charger is: Score(User, charger) = 1/2 (S charger,phone + S charger,phonecover);
* Limitation: does not use - context, user features, product features; Cold start problem - no purchase history - new users, new products;

###### Solution 4: Discovering hidden structure by matrix factorization.
* It takes into consideration - person, their features, product features and finds a sync, also considers the interaction between users and their products.

### Movie Recommendation:
* Table : Users watch movies and rate them;
* From the movie corpus - a matrix is built - cells are divided - watched and unwatched cells based on the information from the watched cells the unwatched cells must be filled and evaluated.
* Describe movie v with the topic-labels(How much action, drama, romance, mystery, etc);
* Describe user u with the topic-label(How much the user likes action, drama, romance, mystery, etc);
* Rating(u,v)(hat) = It is the product of two vectors - Estimate of how a user 'u' will like a movie 'v' given they have never seen it before.
* The fit on the model is the Residual Sum of Squares (RSS);
* *RSS(L, R)= (Observerd (L, R) - Predicted(L, R))^2 + include all (u, v) pairs where Rating Observered(u,v) is available (black squares));
* Limitation (Matrix Factorization - Model still can't deal with new users or movies);

### Featured Matrix Factorization:
* It combines features and discovered topics.
* In case of a new user - uses their features like age, gender, perference, etc to predict items to the users.

### Performance metrics for recommender systems:
* Precision = # liked and shown / # shown; (How much thing i have to look into compared to what i like);
* Recall = # likes and shown / # liked; (How much the recommended items cover the things that i'm interested in)

#### Algorithm:
* Load the dataset - user_id, song_id, listen_count, title, artist, songs; count the number of unique users;
* Create a song recommender - split the data 80% / 20% - training / test;
* Model 1 - Simple popularity recommender -> input=train_data ; feature = userid; target = songid;
* Use the simple popularity model to make recommendations;
* Model 2 - Build a song recommender with personalization -> input=train_data ; feature = userid; target = songid;
* Apply the personalization model to make song recommendations.

-------------------------------
## 5. Deep Learning (Visual Product Recommender)
-------------------------------
* Neural Networks - It provides non-linear representation of data/features; 
* In normal classification model - image classification -> Input : features / image pixel; Output : predicted object;
* Linear classifiers - create a line or linear decision boundary between + / - classes;
* While in Neural networks - classifiers are represented by graphs, here there is a node for each feature - x1, x2, x3, ..., xn; and one output node y(prediction);
###### A "one layer neural network" is similar to the "linear classifier" as both can perform "OR" and "AND" operation. But a simple "linear classifier" cannot perform "XOR" operations, or they cannot segregate data when there is no line that separates the + from the -;
#### Standard Image classification approach:
* Using the hand created image feature - SIFT feature;
* Input : Extract features (create a vector - based on locations where the firing occured) -> Feed it to a simple classifier (logistic regression, SVM - support vector machine); -> detect;
* Challenges - Many hand created features exist for finding interest points - but a painful process to design.

### Deep Learning - Neural Networks - implicitly learn features 
* Neural Networks capture different types of image features at different layers and they get learnt automatically.
* Pros : Enables learning features rather than hand tuning; Impressive performance gains for Computer Vision, Speech recognition, some text analysis;
* Challenges : Requires a lot of data, and labelled data; Splitting training /validation set -> training deep neural network -> time consuming;
* Computationally expensive and extremely hard to tune - choice of architectute, parameter types, hyperparameters, learning algorithm ,etc;

#### Deep Features:
* Deep Features = Deep Learning + Transfer Learning;
* It allows to build neural networks even in the absence of large amount of data.
* Transfer Learning - Use data from one task to help learn in another task;
* Since there are many layers, the last few are relevant to the specific task. Hence the last few layers can be chipped of and the rest can then be feed into other algorithm for attaining predictions.

#### Algorithm:
* Load the image analysis dataset.
* Explore the data - (id, image, label, deep_features, image_array)
* Model 1: Train classifier on raw pixels (features are raw pixels and not deep features); - input : image train data; feature - image_array, target - label; algorith - logistic_classifer;
* Make predictions with the simple model.
* Evaluate the model on the test-data -> accuracy, precision, recall;
* Model 2: Deep Features (Tranfer Learning) - input : image_train, features : deep_features, target : label, algorithm : logistic regression.
* Make predictions with the deep feature model.
* Compute the test data accuracy of the deep_feature model.
