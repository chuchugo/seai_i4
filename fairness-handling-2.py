# Load all necessary packages
import sys
from turtle import pos
sys.path.insert(1, "../")  

import numpy as np
np.random.seed(0)

from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from numpy import mean
from numpy import std
from pandas import Categorical, read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import aif360.datasets as data

from IPython.display import Markdown, display
import pandas as pd


def convertSex(age_cat):
    if age_cat == 'A91' or age_cat =="A94" or age_cat =='A93':
        return 'male'
    else:
        return'female'

def load_dataset(full_path):
    # load the dataset as a numpy array
    dataframe = read_csv(full_path)
    # split into inputs and outputs
    # dataframe.astype({'duration': 'int64'}).dtypes
    dataframe = pd.read_csv("german.csv")
    features_dataframe = dataframe.iloc[:,:-1]
    label_dataframe = dataframe.iloc[:,-1:]

    features_dataframe['Sex_cat'] = 'male'
    features_dataframe.loc[features_dataframe['Sex'] == 'A92', 'Sex_cat'] = 'female'

    interval = (18, 25, 35, 60, 120)

    cats = ['Student', 'Young', 'Adult', 'Senior']
    features_dataframe["Age_cat"] = pd.cut(features_dataframe.Age, interval, labels=cats)

    # features_dataframe = features_dataframe.drop(columns=["Age","Sex"])
    X, y = features_dataframe, label_dataframe
    orig_dataset = pd.merge(
        pd.DataFrame(X).reset_index(),
        pd.DataFrame(y).reset_index(), 
        left_index=True, 
        right_index=True, 
        how="outer")

    orig_dataset = orig_dataset.drop(columns=["index_y", "index_x"])
    orig_dataset["Risk"] = LabelEncoder().fit_transform(orig_dataset["Risk"])
    return orig_dataset

dataset = load_dataset('german.csv')
dataset_orig = data.StandardDataset(dataset,
    label_name='Risk',
    favorable_classes=[1],
    protected_attribute_names=['Sex_cat'],                                                                                                         
    privileged_classes=[lambda x: x == 'male'],     
    features_to_drop=['Sex'],
    categorical_features=['Checking_account', 'credit_history', 'purpose', 'savings_account', 
        'employment_years', 'gurantors', 'Property', 'Installment_plans',
        'housing', 'job', 'telephone', 'foreign_worker','Age_cat']
    )

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

privileged_groups = [{'Sex_cat': 1}]
unprivileged_groups = [{'Sex_cat': 0}]

metric_orig = BinaryLabelDatasetMetric(dataset_orig, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig.mean_difference())

RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf = RW.fit_transform(dataset_orig)

metric_transf = BinaryLabelDatasetMetric(dataset_transf, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
display(Markdown("#### Transformed training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf.mean_difference())

# calculate f2-measure
def f2_measure(y_true, y_pred):
	return fbeta_score(y_true, y_pred, beta=2)
 
# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# define the model evaluation metric
	metric = make_scorer(f2_measure)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1)
	return scores
scale_transf = StandardScaler()
X_train = scale_transf.fit_transform(dataset_transf.features)
y_train = dataset_transf.labels.ravel()
indices = np.arange(1000)
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(dataset_transf.features, dataset_transf.labels,  indices, test_size=0.30, random_state=1, stratify=dataset_transf.labels)
# define model to evaluate
model = RidgeClassifier()
# define the data sampling
pipeline = Pipeline(steps=[('m',model)])
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

test_dataframe = dataset.iloc[indices_test]
test_dataframe = test_dataframe.iloc[:,:-1]

test_dataframe = pd.merge(pd.DataFrame(test_dataframe).reset_index(), pd.DataFrame(y_test).reset_index(), left_index=True, right_index=True, how="outer")
test_dataframe = pd.merge(test_dataframe, pd.DataFrame(y_pred_proba_test).reset_index(), left_index=True, right_index=True, how="outer")
del test_dataframe[test_dataframe.columns[0]]
test_dataframe = test_dataframe.drop(columns=["index", "index_y"])

test_dataframe.to_csv("test_file_fairness2", index=False)
