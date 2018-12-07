# Directing-customers-to-subscription-through-App-behaviour-analysis-
ML project that classifies whether the app user will buy premium user or not

Data: Mock-up dataset based on trends found in real world case studies; 50 000 instances and 11 features (out of which 1 is list of viewed screens) + additional .csv file with top screens

Goal: Predict which users will not subscribe to the paid membership, so that greater marketing efforts can go into trying to “convert” them to paid users

Challenges: Imbalanced classes (solved by under sampling), high correlation among viewed screens (solved by creating funnels and screens counts in order not to lose any information) 

Algorithms: LR regularization L1 and L2, SVM classifier, GB classifier, RF classifier

Measures: Confusion metrics, Area under the ROC curve

Project delivery: Python script executing locally hosted flask api, that takes in raw data, preprocess them, do the predictions and provide downloadable zipped .xlsx file with 3 columns: user identifier, probability of class 1 and predicted class
