The data

The dataset for this assignment is the Breast Cancer Wisconsin. It contains 699 examples described by 9 numeric attributes. There are two classes – class1 and class2. The features are computed from a digitized image of a fine needle aspirate of a breast mass of the subject. Benign breast cancer tumours correspond to class1 and malignant breast cancer tumours correspond to class2.


Input 
(try: python MyClassifier.py breast-cancer-wisconsin-normalised.csv NN param.csv)

program should take 3 command line arguments:
1. The first argument is the path to the data file.
2. The second is the name of the algorithm to be executed or the option for print the pre-processed dataset:
  a. NN for Nearest Neighbour.
  b. LR for Logistic Regression.
  c. NB for Naïve Bayes.
  d. DT for Decision Tree.
  e. BAG for Ensemble Bagging DT.
  f. ADA for Ensemble ADA boosting DT.
  g. GB for Ensemble Gradient Boosting.
  h. RF for Random Forest.
  i. SVM for Linear SVM.
  j. P for printing the pre-processed dataset
3. The third argument is optional, and should only be supplied to algorithms which require parameters, namely NN, BAG, ADA and GB.


Output

For Linear SVM, program should output exactly 4 lines. The first line contains the optimal C value, the second line contains the optimal gamma value, and the third line contains the best cross-validation accuracy score formatted to 4 decimal places using .4f and the fourth line contains the test set accuracy score also formatted to 4 decimal places.

For Random Forest, program should output exactly 5 lines. The first line contains the optimal n_estimators, the second line contains the optimal max_features, the third line contains the optimal max_leaf_nodes, the fourth line contains the best cross validation accuracy score truncated to 4 decimal places using .4f and the fifth line contains the test set accuracy score also truncated to 4 decimal places.
