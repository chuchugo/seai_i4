# %%
#Load the librarys
import pandas as pd #To work with dataset
import numpy as np #Math library
import seaborn as sns #Graph library that use matplot in background
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix #to plot some parameters in seaborn



# %%
#Importing the data
df_credit = pd.read_csv("test_file_fairness2.csv")

# do not need to convert sex
# def convertSex(age_cat):
#     if age_cat == 'A91' or age_cat =="A94" or age_cat =='A93':
#         return 'male'
#     else:
#         return'female'

# df_credit['Sex_cat'] = df_credit['Sex'].apply(convertSex)
# df_credit.loc[:,['Sex_cat','Sex']]

# %%
#seperate age group
# interval = (18, 25, 35, 60, 120)

# cats = ['Student', 'Young', 'Adult', 'Senior']
# df_credit["Age_cat"] = pd.cut(df_credit.Age, interval, labels=cats)

# df_credit.loc[:,["Age","Age_cat"]]

# %% 
# #### load data
#Searching for Missings,type of data and also known the shape of data
print(df_credit.info())

# %% 
# ### Three fairness measures

# #### 1. Anti classifications

#create test data with swaped sex

#df_sex_test is the original data
df_sex_test = df_credit.groupby('Sex_cat').head(100).reset_index(drop=True)
df_sex_test = df_sex_test.sort_values(by=['Sex_cat']).reset_index(drop=True)
#df_sex_test1 is the the data which changed its sex
df_sex_test1 =df_sex_test.copy()
df_sex_test1['Sex_cat'] = df_sex_test1['Sex_cat'].apply(lambda x: 'male' if x == 'female' else 'female')
df_sex_test1

# %%
#create test data with swaped age

#df_age_test is the original data
#df_age_test1 is the the data which changed its age
df_age_test = df_credit.groupby('Age_cat').head(80).reset_index(drop=True)
df_age_test = df_age_test.sort_values(by=['Age_cat']).reset_index(drop=True)

df_age_test1 =df_age_test.copy()

def changeAgeCategory(age_cat):
    if age_cat == 'Student':
        return 'Young'
    if age_cat =='Young':
        return'Adult'
    if age_cat =="Adult":
        return 'Senior'
    if age_cat=='Senior':
        return 'Student'

df_age_test1['Age_cat'] = df_age_test1['Age_cat'].apply(changeAgeCategory)

# %%
import pickle
# load the model
model = pickle.load(open("model.pkl", "rb"))
print("1. Anti classifications")
# to make predictions based on test data , and test1 data and to see if the result are different
df_age_test=df_age_test.iloc[:,:20]
y_pred_age_test = model.predict(df_age_test)
df_age_test1=df_age_test1.iloc[:,:20]
y_pred_age_test1 = model.predict(df_age_test1)
df_age_test['Age_cat'] == df_age_test1['Age_cat']

df = pd.DataFrame()
df['y_pred_age_test']=y_pred_age_test
df['y_pred_age_test1']=y_pred_age_test1
df

age_same_results_ratio= df[df['y_pred_age_test']==df['y_pred_age_test1']].shape[0] / df.shape[0]
age_same_results_ratio
print("age_same_results_ratio")
print(age_same_results_ratio)
#%%
df_sex_test=df_sex_test.iloc[:,:20]
y_pred_sex_test = model.predict(df_sex_test)
df_sex_test1=df_sex_test1.iloc[:,:20]
y_pred_sex_test1 = model.predict(df_sex_test1)
df_sex_test1['Sex_cat']==df_sex_test['Sex_cat']
df = pd.DataFrame()
df['y_pred_sex_test']=y_pred_sex_test
df['y_pred_sex_test1']=y_pred_sex_test1
df
sex_same_results_ratio= df[df['y_pred_sex_test']==df['y_pred_sex_test1']].shape[0] / df.shape[0]
sex_same_results_ratio
print("sex_same_results_ratio")
print(sex_same_results_ratio)

# %%
#  2. group fainess
#  2.1 evaluate fairness between different age groups
print(" 2. group fainess ")

#accept rate for each age group
def get_age_acceptance_rate(df_credit):
    df_good = df_credit.loc[df_credit["0_y"] == 0].groupby('Age_cat').count()
    df_good['0_y']
    df_bad = df_credit.loc[df_credit["0_y"] == 1].groupby('Age_cat').count()
    df_bad['0_y']
    age_acceptance_rate = df_good['0_y']/(df_bad['0_y']+df_good['0_y'])
    return age_acceptance_rate

age_acceptance_rate = get_age_acceptance_rate(df_credit)
age_acceptance_rate
# between different age group, the acceptance ratio varies a lot. Adult and Young have the highest acceptance rate.
print("age_acceptance_rate:")
print(age_acceptance_rate)

# %% [markdown]
# ##### 2.2 evaluate fairness between different sex groups

# %%
#accept rate for each sex group
def get_sex_acceptance_rate(df_credit):
    y1 = df_credit[df_credit["0_y"]== 0]["Sex_cat"].value_counts()
    y2 = df_credit[df_credit["0_y"]== 1]["Sex_cat"].value_counts()
    sex_acceptance_rate =y1/(y1+y2)
    return sex_acceptance_rate

sex_acceptance_rate = get_sex_acceptance_rate(df_credit)
sex_acceptance_rate
print("sex_acceptance_rate:")
print(sex_acceptance_rate)

# %% 
# #### 3. Calculate FP and FN

# 3.1 confusion matrix of each Age group
print("3. confusion matrix of each Age group")
def get_fpr(df_credit):
    df_matrix = pd.DataFrame(confusion_matrix(df_credit["0_x"], df_credit["0_y"], labels=[0, 1])) #labels=["good","bad credit"]
    df_matrix
    FPR=df_matrix.iloc[0,1] /(df_matrix.iloc[0,1] +df_matrix.iloc[0,0] )
    print(FPR)
print("male fpr")
get_fpr(df_credit[df_credit["Sex_cat"]=='male'])
print("female fpr")
get_fpr(df_credit[df_credit["Sex_cat"]=='female'])
#male has higher false positive rate

# %% 
#3.2 confusion matrix of each Age group
print("Student fpr")
get_fpr(df_credit[df_credit["Age_cat"]=='Student'])
print("Young fpr")
get_fpr(df_credit[df_credit["Age_cat"]=='Young'])
print("Adult fpr")
get_fpr(df_credit[df_credit["Age_cat"]=='Adult'])
print("Senior fpr")
get_fpr(df_credit[df_credit["Age_cat"]=='Senior'])
# cats = ['Student', 'Young', 'Adult', 'Senior']
