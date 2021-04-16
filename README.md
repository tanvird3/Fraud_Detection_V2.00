# Fraudulent Transaction Detector

This app detects fraudulent transactions based on a fitted Xtreme Gradient Boosting model (that takes logistic regression as the core model). 
The data is pulled from an IBM DB2 remote database. The model building process can be accessed at https://colab.research.google.com/drive/18dGNJIx5jkqYxjImYzFwOpkzRoXIPY1a#scrollTo=XBzFcBaix9dG . 

## Disclaimer
1. The model is fitted with a synthesized data, thus should only be used for educational purpose.
2. The outcome variable is highly imbalanced, thus model evaluation is extremely critical here.
