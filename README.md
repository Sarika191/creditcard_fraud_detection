Credit Card Fraud Detection - Machine Learning Project

This project is made for detecting fraud transactions using Machine Learning. The goal is to build a model that can predict whether a transaction is fraud or valid based on past data.
________________________________________________________________________________________________________________________________________________________________________
Objective
To create a machine learning model that:
- Learns from past transaction patterns
- Can identify frauds in real-time
- Reduces risks and false alarms
________________________________________________________________________________________________________________________________________________________________________
Dataset
Used dataset: creditcard.csv
It has 284807 transactions with 30 features plus time, amount, and class
Class = 1 means fraud, Class = 0 means normal
(dataset size is large can't be uploaded here)
________________________________________________________________________________________________________________________________________________________________________

Step by step explanation of the project

1. Imported Libraries
Used numpy and pandas for data handling
Used matplotlib and seaborn for visualizations
Used sklearn for training, testing, and evaluation

2. Data Loading
Loaded the dataset using pandas
Checked how many fraud and valid transactions are present

3. Data Visualization
Plotted boxplot to compare transaction amounts for fraud and normal
Plotted heatmap to check correlation between features

4. Model Building
Used RandomForestClassifier from sklearn.ensemble for training the model and predicting results

5. Model Evaluation
Used metrics like:
- Confusion Matrix
- Classification Report
- Matthews Correlation Coefficient
- ROC Curve and AUC Score
________________________________________________________________________________________________________________________________________________________________________
Final Model Performance

Metric                         Observation
Precision                      High (had very few false detect)
Recall (Sensitivity)           High (most fraud cases detected)
F1 Score                       Balanced measure of precision and recall
Matthews Corr. Coefficient     Shows strong model balance
AUC Score                      0.9527 or 95.27 percent
________________________________________________________________________________________________________________________________________________________________________
Visual Outputs
Boxplot shows amount distribution for fraud and valid transactions
Heatmap shows correlation between all features
ROC Curve shows how well the model separates fraud vs valid transactions
________________________________________________________________________________________________________________________________________________________________________
Author
Sarika

GitHub Link
https://github.com/Sarika191/creditcard_fraud_detection/
