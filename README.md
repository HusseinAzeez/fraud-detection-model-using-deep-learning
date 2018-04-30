# Fraud Detection Model Using Deep Learning
In this rutorial I will you through one of the most important applications in machine learning, I will demonstrate how can construct a high-performance model to predict a credit card fraud by using deep learning model.

# Dataset
The datasets contain transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have **492** frauds out of **284,807** transactions. The dataset is highly unbalanced, the positive class (frauds) account for **0.172%** of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. **Features V1, V2, ... V28** are the principal components obtained with PCA, the only features which have not been transformed with PCA are **'Time' and 'Amount'**. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.


# Model overview
In this tutorial I will use a **3 fully-connected layers** + **Dropout**, and **ReLu** as activation function, the first layer contains 200 units, the second also with 200 units, and the third with a single output unit. 
For optimization algorithm, I used **Adam optimization** to optimize the **Accuracy Matrix**.