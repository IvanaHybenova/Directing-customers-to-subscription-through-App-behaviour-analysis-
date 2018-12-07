# Directing-customers-to-subscription-through-App-behaviour-analysis-
ML project that classifies whether the app user will buy premium user or not

Data: Mock-up dataset based on trends found in real world case studies; 50 000 instances and 11 features (out of which 1 is list of viewed screens) + additional .csv file with top screens

Goal: Predict which users will not subscribe to the paid membership, so that greater marketing efforts can go into trying to “convert” them to paid users

Challenges: Imbalanced classes (solved by under sampling), high correlation among viewed screens (solved by creating funnels and screens counts in order not to lose any information) 

Algorithms: LR regularization L1 and L2, SVM classifier, GB classifier, RF classifier

Measures: Confusion metrics, Area under the ROC curve

Project delivery: Python script executing locally hosted flask api, that takes in raw data, preprocess them, do the predictions and provide downloadable zipped .xlsx file with 3 columns: user identifier, probability of class 1 (buing the paid membershim) and predicted class


Files:
EDA.py - Python script that contains exploration of the data
Model.py - Python script that contains data preprocessing, training and tuning the algorithms and saving the pipeline with final model.
flask_predict_api.py - Python scirpt with the application

appdata10.csv - Dataset provided for the project
top_screens.csv - Additional file containing 58 top screens of the app
raw_unseen_data - This is actually hold out set (test.set) after the split I saved for being able to test the app
new_appdata10 - Dataset after the EDA

Instructions:
Download top_screens.csv, raw_unseen_data.csv, all zip files (after unzipping make sure to have final_model.pkl in separate "model" folder created among the under downloaded files) and flask_predict_api.py. 

Through your command line navigate to the folder you are storing these files. Make sure you have python path in your enviroment variables and run command python flask_predict_api.py  

From your browser navigate to your localhost on port 8000. Click on predict_api and then try it out!.
Insert raw_unseen_data and press execute.
After some time scroll down and click on Download the zip.file, which contains the predictions.


