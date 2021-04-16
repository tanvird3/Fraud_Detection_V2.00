# Fraudulent Transaction Detector

This app detects fraudulent transactions based on a fitted Xtreme Gradient Boosting model (that takes logistic regression as the core model). 

## Disclaimer
1. The model is fitted with a synthesized data, thus should only be used for educational purpose
2. The outcome variable is highly imbalanced, thus model evaluation is extremely critical here
3. A single variable (isunrecognizedDevice) can explain the variability of the outcome variable entirely, thus a high chance of wrong prediction is prevalent here