from email import header
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import RidgeClassifier
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def convertSex(age_cat):
    if age_cat == 'A91' or age_cat =="A94" or age_cat =='A93':
        return 'male'
    else:
        return'female'

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    dataframe = read_csv(full_path)
    # split into inputs and outputs
    # dataframe.astype({'duration': 'int64'}).dtypes
    dataframe = pd.read_csv("german.csv")
    features_dataframe = dataframe.iloc[:,:-1]
    label_dataframe = dataframe.iloc[:,-1:]
    features_dataframe['Sex_cat'] = features_dataframe['Sex'].apply(convertSex)
    interval = (18, 25, 35, 60, 120)

    cats = ['Student', 'Young', 'Adult', 'Senior']
    features_dataframe["Age_cat"] = pd.cut(features_dataframe.Age, interval, labels=cats)
    # dataframe = dataframe.iloc[1:,:]
    features_dataframe = features_dataframe.drop(columns=["Age","Sex","Age_cat","Sex_cat"])
    X, y = features_dataframe, label_dataframe

    # select categorical and numerical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X, y, cat_ix, num_ix
 
# # calculate f2-measure
def f2_measure(y_true, y_pred):
	return fbeta_score(np.ravel(y_true), np.ravel(y_pred), beta=2)
 
# # evaluate a model
def evaluate_model(X, y, model):
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define the model evaluation metric
    metric = make_scorer(f2_measure)
    # evaluate model
    scores = cross_val_score(model, X, np.ravel(y), scoring=metric, cv=cv, n_jobs=-1)
    return scores
 
# define the location of the dataset
full_path = 'german.csv'
# load the dataset
X, y, cat_ix, num_ix = load_dataset(full_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1, stratify=y)
# define model to evaluate
model = RidgeClassifier()
# define the data sampling
sampling = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='majority'))
# one hot encode categorical, normalize numerical
ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',StandardScaler(),num_ix)])
# scale, then sample, then fit model
pipeline = Pipeline(steps=[('t',ct), ('s', sampling), ('m',model)])
# evaluate the model and store results
scores = evaluate_model(X_train, np.ravel(y_train), pipeline)
print('Cross Validation Score %.3f (%.3f)' % (mean(scores), std(scores)))

pipeline.fit(X_train,np.ravel(y_train))
roc_train = roc_auc_score(y_train, pipeline.predict(X_train))
print("ROC score on Train Data :", roc_train)
roc_test = roc_auc_score(y_test, pipeline.predict(X_test))
print("ROC score on Test Data :", roc_test)

y_pred_proba = pipeline.predict(X_train)
fpr_train, tpr_train, _ = metrics.roc_curve(y_train,  y_pred_proba)

#create ROC curve
plt.plot(fpr_train,tpr_train)
plt.ylabel('True Positive Train Rate')
plt.xlabel('False Positive Train Rate')
plt.show()

y_pred_proba_test = pipeline.predict(X_test)
fpr_test, tpr_test, _ = metrics.roc_curve(y_test,  y_pred_proba_test)

#create ROC curve
plt.plot(fpr_test,tpr_test)
plt.ylabel('True Positive Test Rate')
plt.xlabel('False Positive Test Rate')
plt.show()

# frames = [pd.DataFrame(X_test), pd.DataFrame(y_test), pd.DataFrame(y_pred_proba_test)]
# test_dataframe = pd.concat(frames)
test_dataframe = pd.merge(pd.DataFrame(X_test).reset_index(), pd.DataFrame(y_test).reset_index(), left_index=True, right_index=True, how="outer")
test_dataframe = pd.merge(test_dataframe, pd.DataFrame(y_pred_proba_test).reset_index(), left_index=True, right_index=True, how="outer")
del test_dataframe[test_dataframe.columns[0]]
test_dataframe = test_dataframe.drop(columns=["index", "index_y"])

test_dataframe.to_csv("test_file_fairness1.csv", index=False)
