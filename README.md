# Default-Detection

In this project we have tried to estimate the probability that a customer will default or not on repaying their credit. The data that we've used come from the UCI data set found here: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients. 

An accompanying paper on the data collection can be found at:
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

Predicting the probabilty that a customer will default is a classic machine learning classification task. We tested various forms of logistic, SVC, KNN, and decision tree models and ran them over a combination of biographical and finacial data. Ultimately we wanted to find the best classifier model given our very limited time constraints, and so favoured slightly faster, less brute force methods. 

We also briefly investigated the affect of SMOTE vs inbuilt resampling methods on a models predictive power.
