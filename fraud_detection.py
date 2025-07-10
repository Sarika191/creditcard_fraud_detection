import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc
)

# we will be loading the dataset first
print("Loading the dataset ")
data = pd.read_csv('creditcard.csv')
print("Data has been loaded successfully")

# now we will be checking data which are actual valid and which are fraud
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

print(f"\nOutlier Fraction is: {len(fraud) / float(len(valid)):.6f}")
print(f"Total Fraud Transactions: {len(fraud)}")
print(f"Total Valid Transactions: {len(valid)}")

#now we will compare the overall data btw fraud and valid
print("\nTransaction amounts varies for fraud vs normal ones.")
plt.figure(figsize=(8, 4))
sns.boxplot(x='Class', y='Amount', data=data)
plt.title('Amount Distribution by Transaction Type')
plt.show()

#now will create heatmap to see  the feature correlation
print("\nNow plotting the correlation heatmap of features...")
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), cmap='coolwarm', linewidths=0.3)
plt.title("Correlation Between Features")
plt.show()

# now we will be splitting the data for model training 
print("\n Preparing for model training")
X = data.drop('Class', axis=1)
Y = data['Class']

xTrain, xTest, yTrain, yTest = train_test_split(X.values, Y.values, test_size=0.2, random_state=42)

print("Ready to go with Data...")
print(f"Training Data:{len(xTrain)} rows")
print(f"Testing Data:{len(xTest)} rows")

# ok, so now we are training the model
print("\nTraining")
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)
yPred = rfc.predict(xTest)
yProba = rfc.predict_proba(xTest)[:, 1]
print("Training Completed !!")

# based of everthing now we will be evaluating the overall data and performance
print("\nEvaluating now")
print("Confusion Matrix:")
print(confusion_matrix(yTest, yPred))

print("\nHere’s the classification report:")
print(classification_report(yTest, yPred, digits=4))

mcc = matthews_corrcoef(yTest, yPred)
print(f"Matthews Correlation Coefficient: {mcc:.4f}")

# ROC curve, this will let us know our model performance like how good it able to 
# distinguish btw fraud transaction and valid ones
print("\nROC Curve")
fpr, tpr, _ = roc_curve(yTest, yProba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Fraud Detection')
plt.legend()
plt.grid()
plt.show()
input("\nPress Enter to exit (^///^)")
