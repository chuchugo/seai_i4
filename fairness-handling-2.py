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
from pandas import read_csv
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
from sklearn.metrics import confusion_matrix

from IPython.display import Markdown, display
import pandas as pd

dataset_orig = GermanDataset(
    protected_attribute_names=['age'],           # this dataset also contains protected
                                                 # attribute for "sex" which we do not
                                                 # consider in this evaluation
    privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
    features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
)

dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

privileged_groups = [{'age': 1}]
print(privileged_groups)
unprivileged_groups = [{'age': 0}]

metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(dataset_orig_train)

metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
display(Markdown("#### Transformed training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())

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

X_train, X_test, y_train, y_test = train_test_split(dataset_transf_train.features, dataset_transf_train.labels, test_size=0.30, random_state=1, stratify=dataset_transf_train.labels)
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
fpr_train, tpr_train, _ = metrics.roc_curve(y_train,  y_pred_proba, pos_label=2)

#create ROC curve
plt.plot(fpr_train,tpr_train)
plt.ylabel('True Positive Train Rate')
plt.xlabel('False Positive Train Rate')
plt.show()

y_pred_proba_test = pipeline.predict(X_test)
fpr_test, tpr_test, _ = metrics.roc_curve(y_test,  y_pred_proba_test, pos_label=2)

#create ROC curve
plt.plot(fpr_test,tpr_test)
plt.ylabel('True Positive Test Rate')
plt.xlabel('False Positive Test Rate')
plt.show()

test_dataframe = pd.merge(pd.DataFrame(X_test).reset_index(), pd.DataFrame(y_test).reset_index(), left_index=True, right_index=True, how="outer")
test_dataframe = pd.merge(test_dataframe, pd.DataFrame(y_pred_proba_test).reset_index(), left_index=True, right_index=True, how="outer")
del test_dataframe[test_dataframe.columns[0]]
test_dataframe = test_dataframe.drop(columns=["index", "index_y"])

test_dataframe.to_csv("test_file_fairness2.csv", index=False)
