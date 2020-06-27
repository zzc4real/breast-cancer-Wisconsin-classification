import numpy as np
import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from numpy import nan as NA
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
folds_num = 10


def change_to_nan(x):
    if x == '?':
        x = NA
    return x


def change_from_class_to_num(x):
    if x == 'class1':
        x = 0
    elif x == 'class2':
        x = 1
    return x


def preprocessed_output(data):
    data.to_csv('prepared_data.txt', header=None, index=False)
    file = open('prepared_data.txt')
    print(file.read(), end='')


def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return nsum / len(num)


def kNNClassifier(x, y, k):
    from sklearn.neighbors import KNeighborsClassifier
    scores = []
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=0)

    for i, j in skf.split(x, y):
        x_train = x[i]
        x_test = x[j]
        y_train = y[i]
        y_test = y[j]
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(x_train, y_train)
        scores.append(clf.score(x_test, y_test))

    if(len(scores)) == folds_num:
        return averagenum(scores)
    else:
        print("algorithm error in KNN classifier.", scores)
        return -1


def bagDTClassifier(x, y, ne, ms, md):
    from sklearn.ensemble import BaggingClassifier

    scores = []
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=0)
    for i, j in skf.split(x, y):
        x_train = x[i]
        x_test = x[j]
        y_train = y[i]
        y_test = y[j]
        clf = BaggingClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=md), n_estimators=ne, max_samples=ms, bootstrap=True, random_state=0)
        clf.fit(x_train, y_train)
        scores.append(clf.score(x_test, y_test))

    if (len(scores)) == folds_num:
        return averagenum(scores)
    else:
        print("algorithm error in KNN classifier.", scores)
        return -1


def adaDTClassifier(x, y, ne, lr, md):
    from sklearn.ensemble import AdaBoostClassifier

    scores = []
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=0)
    for i, j in skf.split(x, y):
        x_train = x[i]
        x_test = x[j]
        y_train = y[i]
        y_test = y[j]
        clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', max_depth=md), n_estimators=ne, learning_rate=lr, random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        scores.append(accuracy_score(y_test, y_pred))

    if (len(scores)) == folds_num:
        return averagenum(scores)
    else:
        print("algorithm error in KNN classifier.", scores)
        return -1


def gbClassifier(x, y, ne, lr):
    from sklearn.ensemble import GradientBoostingClassifier
    scores = []
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=0)
    for i, j in skf.split(x, y):
        x_train = x[i]
        x_test = x[j]
        y_train = y[i]
        y_test = y[j]
        clf = GradientBoostingClassifier(n_estimators=ne, learning_rate=lr, random_state=0)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        scores.append(accuracy_score(y_test, y_pred))

    if (len(scores)) == folds_num:
        return averagenum(scores)
    else:
        print("algorithm error in KNN classifier.", scores)
        return -1


def logregClassifier(x, y):
    scores = []
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=0)
    from sklearn.linear_model import LogisticRegression
    for i, j in skf.split(x, y):
        x_train = x[i]
        x_test = x[j]
        y_train = y[i]
        y_test = y[j]
        clf = LogisticRegression()
        clf.fit(x_train, y_train)
        scores.append(clf.score(x_test, y_test))

    if (len(scores)) == folds_num:
        return averagenum(scores)
    else:
        print("algorithm error in KNN classifier.", scores)
        return -1


def nbClassifier(x, y):
    scores = []
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=0)
    from sklearn.naive_bayes import GaussianNB
    for i, j in skf.split(x, y):
        x_train = x[i]
        x_test = x[j]
        y_train = y[i]
        y_test = y[j]
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        scores.append(accuracy_score(y_test, y_pred))

    if (len(scores)) == folds_num:
        return averagenum(scores)
    else:
        print("algorithm error in KNN classifier.", scores)
        return -1


def dtClassifier(x, y):
    scores = []
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=0)
    for i, j in skf.split(x, y):
        x_train = x[i]
        x_test = x[j]
        y_train = y[i]
        y_test = y[j]
        clf = DecisionTreeClassifier(criterion='entropy',random_state=0)
        clf.fit(x_train, y_train)
        scores.append(clf.score(x_test, y_test))

    if (len(scores)) == folds_num:
        return averagenum(scores)
    else:
        print("algorithm error in KNN classifier.", scores)
        return -1


def bestRFClassifier(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)
    param_grid = {'n_estimators': [10, 20, 50, 100],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_leaf_nodes': [10, 20, 30]}
    from sklearn.ensemble import RandomForestClassifier
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=0)
    grid_search = GridSearchCV(RandomForestClassifier(random_state=0, criterion='entropy'), param_grid, cv=skf,
                               return_train_score=True)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_['n_estimators'])
    print(grid_search.best_params_['max_features'])
    print(grid_search.best_params_['max_leaf_nodes'])
    print("{:.4f}".format(grid_search.best_score_))
    return grid_search.score(x_test, y_test)


def bestLinClassifier(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, random_state=0)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
    from sklearn.svm import SVC
    skf = StratifiedKFold(n_splits=folds_num, shuffle=True, random_state=0)
    grid_search = GridSearchCV(SVC(kernel="linear"), param_grid, cv=skf,
                               return_train_score=True)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_['C'])
    print(grid_search.best_params_['gamma'])
    print("{:.4f}".format(grid_search.best_score_))
    return grid_search.score(x_test, y_test)


def main(argv):
    data_path = ""
    clf_name = ""
    param_path = ""
    if len(argv) == 3:
        data_path = argv[1]
        clf_name = argv[2]
    elif len(argv) == 4:
        data_path = argv[1]
        clf_name = argv[2]
        param_path = argv[3]

    # load target data and change class1 to num
    data = pd.read_csv(data_path)
    data['class'] = data['class'].map(change_from_class_to_num)

    # change ? to nan
    data = data.applymap(change_to_nan)
    # print(data[0:24]["Bare Nuclei"])

    # imputation
    imputer = SimpleImputer(NA, "mean")
    data_imputation = imputer.fit_transform(data)

    # normalization
    minmax_transform = MinMaxScaler(feature_range=(0, 1))
    data_normalization = minmax_transform.fit_transform(data_imputation)
    data_normalization = pd.DataFrame(data_normalization, dtype=np.float)

    # formatted to 4 decimal places
    format = lambda x: '%.4f' % x
    attribute_data = data_normalization.iloc[:,:-1]
    attribute_data = attribute_data.applymap(format)
    class_data = pd.DataFrame(data_normalization.iloc[:,-1:], dtype=np.int)
    pp_data = attribute_data.join(class_data)


    ############ classification algorith with 10 folds ############
    # read parameter from param.csv
    param_df = []
    if param_path != '':
        param_df = pd.read_csv(param_path)
        # param = param_df.loc[0].values.tolist()
    avg_cvs = NA

    # core
    # transfer from dataframe to array
    attribute = attribute_data.values
    class_d = class_data.values.T
    class_d = class_d[0]

    if clf_name == "NN":
        k_value = NA
        if len(param_df.loc[0]) >= 1:
            k_value = int(param_df.at[0, 'K'])
            avg_cvs = kNNClassifier(attribute, class_d, k_value)
        else:
            print("did not find parameter k")

    elif clf_name == "LR":
        avg_cvs = logregClassifier(attribute, class_d)

    elif clf_name == "NB":
        avg_cvs =  nbClassifier(attribute, class_d)
    elif clf_name == "DT":
        avg_cvs = dtClassifier(attribute, class_d)

    elif clf_name == "BAG":
        n_estimators = NA
        max_samples = NA
        max_depth = NA
        if len(param_df.loc[0]) >= 3:
            n_estimators = int(param_df.at[0, 'n_estimators'])
            max_samples = int(param_df.at[0, 'max_samples'])
            max_depth = int(param_df.at[0, 'max_depth'])
            avg_cvs = bagDTClassifier(attribute, class_d, n_estimators, max_samples, max_depth)
        else:
            print(" bagDTClassifier parameters are incorrect")

    elif clf_name == "ADA":
        n_estimators = NA
        learning_rate = NA
        max_depth = NA
        if len(param_df.loc[0]) >= 3:
            n_estimators = int(param_df.at[0, 'n_estimators'])
            learning_rate = float(param_df.at[0, 'learning_rate'])
            max_depth = int(param_df.at[0, 'max_depth'])
            avg_cvs = adaDTClassifier(attribute, class_d, n_estimators, learning_rate, max_depth)
        else:
            print(" adaDTClassifier parameters are incorrect")

    elif clf_name == "GB":
        n_estimators = NA
        learning_rate = NA
        if len(param_df.loc[0]) >= 2:
            n_estimators = int(param_df.at[0, 'n_estimators'])
            learning_rate = float(param_df.at[0, 'learning_rate'])
            avg_cvs = gbClassifier(attribute, class_d, n_estimators, learning_rate)
        else:
            print(" adaDTClassifier parameters are incorrect")

    elif clf_name == "RF":
        avg_cvs = bestRFClassifier(attribute, class_d)

    elif clf_name == "SVM":
        avg_cvs = bestLinClassifier(attribute, class_d)
    
    elif clf_name == "P":
        # print preprocessed data
        preprocessed_output(pp_data)

    if clf_name != "P":
        print('%.4f' % avg_cvs)

if __name__ == "__main__":
    main(sys.argv)
