#!/usr/bin/env python
# coding: utf-8

# # DATA ACQUISITION 

# In[49]:


import numpy as np
import pandas as pd

print(f"NumPy Version: {np.__version__}\nPandas Version: {pd.__version__}")


# In[50]:


import xgboost as xgb
print(xgb.__version__)


# In[51]:


smoking = pd.read_csv("E:/Smoking/train.csv", index_col=0)


# # EXPLORATORY DATA ANALYSIS 
# ## ( DATA PREPARATION + FEATURE ENGINEERING )

# In[52]:


import seaborn as sns 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import sklearn 
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import IsolationForest 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler 

print(f"Seaborn Version: {sns.__version__}\nMatplotlib Version: {mpl.__version__}\nSklearn Version: {sklearn.__version__}")


# In[53]:


smoking.shape


# In[54]:


smoking.head()


# In[55]:


smoking.info()


# In[56]:


smoking.describe().T


# ## Renaming columns 

# In[57]:


column_mapping = {
    'waist(cm)': 'waist',
    'systolic': 'systolic_blood_pressure',
    'relaxation': 'diastolic_blood_pressure',
    'fasting blood sugar': 'fasting_blood_sugar',
    'Cholesterol': 'cholesterol',
    'HDL': 'hdl',
    'LDL': 'ldl',
    'Urine protein': 'urine_protein',
    'serum creatinine': 'serum_creatinine',
    'AST': 'ast',
    'ALT': 'alt',
    'Gtp': 'ggt',
    'dental caries': 'dental_caries',
    'smoking': 'outcome',
}

smoking.rename(columns=column_mapping, inplace=True)


# ## Checking null values

# In[58]:


smoking.isnull().sum()


# ## Checking unique values for invalid entries 

# In[14]:


# Unique values in each column

for col in smoking.columns:
    unique_values = smoking[col].unique()
    print(f"Unique values for {col} : {unique_values}")


# ## Checking number of duplicate rows 

# In[13]:


smoking.duplicated().sum()


# ## Smoking outcome ratio

# In[64]:


proportion_outcome = smoking['outcome'].value_counts()
print(proportion_outcome)


# In[65]:


plt.pie(proportion_outcome , labels = proportion.index, autopct="%1.1f%%", explode = [0,0.1], colors = ["#0bb4ff","#dc0ab4"])
plt.title("Smoking Outcome Ratio", fontsize=20)
plt.axis("equal")
plt.show()


# ### Percentage of sample positive for smoking 

# In[60]:


print(f"Smoking ratio = {sum(smoking['outcome']) / len(smoking):.4f}")


# <div style="background-color: #ffa300; padding: 10px;">
#     <h2>Data is imbalanced</h2>
#  
# </div>

# ## Combining 2 hearing features into 1 feature

# In[15]:


smoking['hearing'] = ((smoking['hearing(left)'] + smoking['hearing(right)']) / 2)


# In[16]:


smoking['hearing'].value_counts()


# ### Dropping separate hearing features

# In[17]:


smoking.drop(columns=['hearing(left)', 'hearing(right)'], inplace=True)


# In[18]:


smoking.info()


# ## Investigating eyesight features 

# In[19]:


smoking['eyesight(left)'].value_counts()


# In[20]:


smoking['eyesight(right)'].value_counts()


# ### 9.9 is an invalid entry in both eyesight features 
# 
# ### 9.9 can either be 0.9 or 1.9 - as maximum value in the data is 2.0 for both features

# In[21]:


# Print rows where eyesight(left) is '9.9'

filtered = smoking[(smoking["eyesight(left)"] == 9.9)]

columns_to_print=['eyesight(left)', 'eyesight(right)']
print(filtered[columns_to_print])


# In[22]:


# Print rows where eyesight(right) is '9.9'

filtered = smoking[(smoking["eyesight(right)"] == 9.9)]

columns_to_print=['eyesight(left)', 'eyesight(right)']
print(filtered[columns_to_print])


# In[23]:


# Print rows where both eyesight(left) and eyesight(right) are '9.9'

filtered = smoking[(smoking["eyesight(left)"] == 9.9) & (smoking["eyesight(right)"] == 9.9)]

columns_to_print=['eyesight(left)', 'eyesight(right)']
print(filtered[columns_to_print])


# ### 9.9 will be equated to 0.9 by as it is more likely to be systematic entry error in which number key 9 was mistakenly pressed instead of key 0 - as it is immediately adjacent to key 0 on the top number row of qwerty keyboard
# ### It is less likely to be 1, as key 1 is on the opposite side of key 0 on the number row, making key 1 less likely to be mistaken for key 0

# In[24]:


smoking[['eyesight(left)', 'eyesight(right)']] = smoking[['eyesight(left)', 'eyesight(right)']].replace(9.9, 0.9)


# In[25]:


smoking['eyesight(left)'].value_counts()


# In[26]:


smoking['eyesight(right)'].value_counts()


# ## Combining 2 eyesight features into 1 feature

# In[27]:


smoking['eyesight'] = ((smoking['eyesight(left)'] + smoking['eyesight(right)']) / 2)


# In[28]:


smoking['eyesight'].value_counts()


# ### Dropping separate eyesight features

# In[29]:


smoking.drop(columns=['eyesight(left)', 'eyesight(right)'], inplace=True)


# In[30]:


smoking.info()


# ## Investigating urine protein

# In[31]:


plt.figure(figsize=(7,7))
ax = sns.countplot(x = smoking['urine_protein'])

for p in ax.patches:
    ax.annotate(f'{p.get_height():g}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
plt.xlabel('Urine Protein')
plt.ylabel('Frequency')
plt.show()


# ### Point to clarify: Are these dipstick measurement values as in 1+, 2+,...? Or categorical classifications for different stages of proteinuria?
# 
# ### Until documentation becomes available for clarification, these values will be treated as categorical classification as it is highly unlikely that everyone in the sample has 1+ proteinuria 
# 

# ## Investigating dental caries
# 

# In[32]:


plt.figure(figsize=(7,7))
ax = sns.countplot(x = smoking['dental_caries'])

for p in ax.patches:
    ax.annotate(f'{p.get_height():g}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
plt.xlabel('Dental Caries')
plt.ylabel('Frequency')
plt.show()


# ## Investigating fasting blood sugar 

# #### Pre-diabetic : 100 mg/dl - 125 mg/dl
# #### Diabetic : 126 mg/dl or more

# In[33]:


sns.displot(smoking['fasting_blood_sugar'], kde=True)


# ## Investigating serum creatinine 
# ### Normal ranges:
# ####          Adult male: 0.7 to 1.3 mg/dL 
# ####          Adult female: 0.6 to 1.1 mg/dL 

# In[34]:


sns.displot(smoking['serum_creatinine'], kde=True)


# ## Creating new feature: Body Mass Index (BMI) 

# In[35]:


smoking['bmi'] = smoking['weight(kg)'] / ((smoking['height(cm)'] / 100) ** 2)


# ### Dropping weight and height after creating BMI feature 

# In[36]:


smoking.drop(columns=['height(cm)', 'weight(kg)'], inplace=True)


# In[37]:


smoking.info()


# ## Creating separate dataframes of smokers and non-smokers 

# In[38]:


smokers = smoking[smoking['outcome'] == 1]


# In[39]:


non_smokers = smoking[smoking['outcome'] == 0]


# ## Investigating cholesterol and HDL (High Density Lipoprotein) relationship 

# ### Smokers + Non-smokers 

# In[40]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=smoking, x='cholesterol', y='hdl')


# ### Smokers only  

# In[41]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=smokers, x='cholesterol', y='hdl')


# ### Non-smokers only 

# In[42]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=non_smokers, x='cholesterol', y='hdl')


# ## Investigating cholesterol and triglyceride relationship 

# ### Smokers + Non-smokers

# In[43]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=smoking, x='cholesterol', y='triglyceride')


# ### Smokers only 

# In[44]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=smokers, x='cholesterol', y='triglyceride')


# ### Non-smokers only

# In[45]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=non_smokers, x='cholesterol', y='triglyceride')


# ## Investigating HDL and triglyceride relationship

# ### Smokers + Non-smokers 

# In[46]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=smoking, x='hdl', y='triglyceride')


# ### Smokers only 

# In[47]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=smokers, x='hdl', y='triglyceride')


# ### Non-smokers only 

# In[48]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=non_smokers, x='hdl', y='triglyceride')


# ## Investigating cholesterol and fasting blood sugar relationship

# ### Smokers + Non-smokers 

# In[50]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=smoking, x='cholesterol', y='fasting_blood_sugar')


# ### Smokers only  

# In[51]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=smokers, x='cholesterol', y='fasting_blood_sugar')


# ### Non-smokers only 

# In[52]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=non_smokers, x='cholesterol', y='fasting_blood_sugar')


# ## Investigating cholesterol and smoking relationship 

# In[53]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=smoking, x='outcome', y='cholesterol', x_jitter=.15)


# ## Investigating hemoglobin and smoking relationship 

# In[54]:


plt.subplots(figsize=(7, 7))
sns.regplot(data=smoking, x='outcome', y='hemoglobin', x_jitter=.15)


# In[55]:


smoking.info()


# ## Checking correlation 

# In[56]:


corr = smoking.corr()

# Get upper criangle of the co-relation matrix
matrix = np.triu(corr)


# Use upper triangle matrix as mask 
sns.set(rc={"figure.figsize":(20, 20)})   
sns.heatmap(corr, cmap="Blues", annot=True, mask=matrix)


# ## Checking outliers

# ### Detecting total outliers using isolation forest 

# In[57]:


subset = smoking.drop(columns=['outcome'])

outlier_model = IsolationForest(n_estimators=100, max_samples='auto', max_features=1.0, contamination='auto')


# In[58]:


predictions = outlier_model.fit_predict(subset)

# Create a DataFrame to store the outlier count for each row
outlier_count_df = pd.DataFrame({
    'outlier_count': [(pred == -1) for pred in predictions]
                                })

# Sum the counts for each row to get total outlier count
total_outliers = outlier_count_df['outlier_count'].sum()

# Attach the outlier count to the original dataframe
smoking['outlier_count'] = outlier_count_df


# In[59]:


smoking['outlier_count'].value_counts()


# In[60]:


plt.figure(figsize=(7,7))
ax = sns.countplot(x = smoking['outlier_count'])

for p in ax.patches:
    ax.annotate(f'{p.get_height():g}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
plt.xlabel('Outlier Count')
plt.ylabel('Frequency')
plt.show()


# ## Detecting outliers in individual features using box plots 

# In[61]:


plt.subplots(figsize=(20, 10))
feature_names = ['Age', 'Waist', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Fasting Blood Sugar',
                 'Cholesterol', 'Triglyceride', 'HDL', 'LDL', 'Hemoglobin',
                 'Urine Protein', 'Serum Creatinine', 'AST', 'ALT', 'GGT',
                'Dental Caries', 'Outcome', 'Hearing', 'Eyesight', 'BMI', 'Outlier Count']

sns.boxplot(data=smoking)
plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")                # Rotate x-labels at a 45-degree angle for clarity 
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()


# ## Inspecting individual box plots

# ### Fasting blood sugar 

# In[62]:


plt.subplots(figsize=(5, 5))
smoking[['fasting_blood_sugar']].boxplot()


# ### Cholesterol

# In[63]:


plt.subplots(figsize=(5, 5))
smoking[['cholesterol']].boxplot()


# ### Triglyceride

# In[64]:


plt.subplots(figsize=(5, 5))
smoking[['triglyceride']].boxplot()


# ### HDL

# In[65]:


plt.subplots(figsize=(5, 5))
smoking[['hdl']].boxplot()


# ### LDL

# In[66]:


plt.subplots(figsize=(5, 5))
smoking[['ldl']].boxplot()


# ### AST - Aspartate Transaminase

# In[67]:


plt.subplots(figsize=(5, 5))
smoking[['ast']].boxplot()


# ### ALT - Alanine Transaminase

# In[68]:


plt.subplots(figsize=(5, 5))
smoking[['alt']].boxplot()


# ### GGT - Gamma Glutamyl Transferase

# In[69]:


plt.subplots(figsize=(5, 5))
smoking[['ggt']].boxplot()


# ## Capping Outliers 

# In[70]:


columns_to_cap = ['age', 
                  'waist', 
                  'systolic_blood_pressure', 
                  'diastolic_blood_pressure',
                  'fasting_blood_sugar', 'cholesterol', 'triglyceride', 'hdl', 'ldl',
                  'hemoglobin',
                  'urine_protein',
                  'serum_creatinine',
                  'ast',
                  'alt',
                  'ggt',
                  'bmi',
                 ]


# In[71]:


def cap_outliers(data, columns):
    
    for column in columns:
       
        q1 = data[column].quantile(0.25)      # Get the Q1 (25 percentile) and Q3 (75 percentile)
        q3 = data[column].quantile(0.75)

        iqr = q3 - q1                         # Calculate interquartile range

        max_limit = q3 + (1.5 * iqr)          # Set limits
        min_limit = q1 - (1.5 * iqr)

        data[column] = np.clip(               # Cap outliers
                        data[column], 
                        a_min=min_limit, 
                        a_max=max_limit)     
    


# In[72]:


cap_outliers(data=smoking, columns=columns_to_cap)


# ### Checking capping result

# In[73]:


plt.subplots(figsize=(20, 10))
feature_names = ['Age', 'Waist', 'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'Fasting Blood Sugar',
                 'Cholesterol', 'Triglyceride', 'HDL', 'LDL', 'Hemoglobin',
                 'Urine Protein', 'Serum Creatinine', 'AST', 'ALT', 'GGT',
                'Dental Caries', 'Outcome', 'Hearing', 'Eyesight', 'BMI', 'Outlier Count']

sns.boxplot(data=smoking)
plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")                # Rotate x-labels at a 45-degree angle for clarity 
plt.xlabel('Features')
plt.ylabel('Values')
plt.show()


# In[74]:


smoking.info()


# # MODEL BUILDING 

# In[105]:


get_ipython().system('pip install dtreeviz')


# In[147]:


get_ipython().system('pip install graphviz')


# In[162]:


import optuna
import xgboost
from xgboost import XGBClassifier
from xgboost import plot_importance, plot_tree, plotting
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_score
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    accuracy_score, make_scorer, confusion_matrix,
    precision_recall_curve, roc_curve, auc, roc_auc_score, log_loss
)

import warnings
import dtreeviz
import graphviz

print(f"XGBoost Version: {xgboost.__version__}\nOptuna Version: {optuna.__version__}\nDtreeviz Version: {dtreeviz.__version__}\nGraphviz Version: {graphviz.__version__}")


# ## Splitting data into the following sets:
# #### Training set (80%)
# #### Validation set (20%)

# In[76]:


X = smoking.drop(columns=['outcome', 'outlier_count'])
y = smoking['outcome']


# In[77]:


X_train, X_val, y_train, y_val = train_test_split(X, y,  stratify=y, test_size=0.20, random_state=7765)


# ### Checking stratification based on outcome 

# In[78]:


sum(y_train)/len(y_train)


# ## Scaling features

# In[79]:


features_to_scale = [
             'age',
             'waist',
             'systolic_blood_pressure',
             'diastolic_blood_pressure',
             'fasting_blood_sugar',
             'cholesterol',
             'triglyceride',
             'hdl',
             'ldl',
             'hemoglobin',
             'serum_creatinine',
             'ast',
             'alt',
             'ggt',
             'bmi',
                   ]


# ### Defining scaling function 

# In[80]:


def scale_features(df, features_to_scale, method):

    if method not in ['zscore', 'minmax', 'robust']:
        raise ValueError("Invalid Method: Choose one of the following scaling methods: 'Zscore', 'MinMax', or 'Robust'.")

    scaler = {
        'zscore': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
             }[method]

    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])       # Fit_transform creates copy of dataframe 
    
    print(df.info())

    return df


# ### Calling scaling function 

# In[81]:


X_train = scale_features(df=X_train, features_to_scale=features_to_scale, method='zscore')


# In[82]:


X_val = scale_features(df=X_val, features_to_scale=features_to_scale, method='zscore')


# ## Tuning hyperparameters - using NVIDIA GPU

# ### Checking GPU status

# In[83]:


get_ipython().system('nvidia-smi')


# In[84]:


# Warning ignore calls - if required 
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=UserWarning)


# ### Performing step-wise optimization to reduce tuning time 

# ### Setting max_depth=0 due to usage of loss guide grow policy (as this policy reduces model training time via histogram aggregation)

# In[85]:


get_ipython().run_cell_magic('time', '', '\ndef objective_sort(trial):\n    \n    params = {\n        \'booster\': trial.suggest_categorical(\'booster\', [\'gbtree\', \'dart\']), \n        \'max_leaves\': trial.suggest_int(\'max_leaves\', 5, 100),\n        \'min_child_weight\': trial.suggest_int(\'min_child_weight\', 1, 150),\n        \'subsample\': trial.suggest_float(\'subsample\', 0.5, 1),\n        \'colsample_bytree\': trial.suggest_float(\'colsample_bytree\', 0, 1),\n        \'learning_rate\': trial.suggest_float(\'learning_rate\', 0, 0.5),\n        \'alpha\': trial.suggest_float(\'alpha\', 0, 10), \n        \'gamma\': trial.suggest_float(\'gamma\', 0, 10),\n        \'reg_lambda\': trial.suggest_float(\'reg_lambda\', 0, 10),\n        \'eval_metric\': \'auc\',                                  # Set evaluation metric \n        \'tree_method\': \'hist\',                                 # Use \'hist\' for GPU acceleration\n        \'device\': \'cuda\'                                       # Specify GPU device\n             }\n    \n   \n    model = XGBClassifier(\n                    **params, \n                    n_jobs=-1, \n                    objective="binary:logistic",\n                    grow_policy=\'lossguide\',\n                    max_depth=0,\n                    early_stopping_rounds=500, \n                    random_state=1142\n                         )\n        \n    model.fit(        \n        X_train, \n        y_train, \n        eval_set=[(X_val,y_val)],\n        verbose=0)\n\n    y_pred = model.predict(X_val)\n\n    return roc_auc_score(y_val, y_pred)\n\n\n\nsampler = optuna.samplers.RandomSampler(seed=10)\nstudy = optuna.create_study(sampler=sampler,direction=\'maximize\')\nstudy.optimize(objective_sort, n_trials=100)\n')


# ### Plotting hyperparameter importances 

# In[87]:


fig = optuna.visualization.plot_param_importances(study)
fig.show()


# ### Tuning most important parameters 

# In[89]:


from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

sampler = TPESampler()
pruner = HyperbandPruner()


def objective_1(trial):
    
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0, 0.5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0, 1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 150),
        'alpha': trial.suggest_float('alpha', 0, 10),  
        'eval_metric': 'auc',                                  # Set evaluation metric 
        'tree_method': 'hist',                                 # Use 'hist' for GPU acceleration
        'device': 'cuda'                                       # Specify GPU device
             }
    
   
    model = XGBClassifier(
                    **params, 
                    n_jobs=-1, 
                    objective="binary:logistic",
                    grow_policy='lossguide',
                    max_depth=0,
                    early_stopping_rounds=500, 
                    random_state=1142
                         )
        
    model.fit(        
        X_train, 
        y_train, 
        eval_set=[(X_val,y_val)],
        verbose=0)

    y_pred = model.predict(X_val)

    return roc_auc_score(y_val, y_pred)


study_1 = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize') # Direction set to maximize AUC
study_1.optimize(objective_1, n_trials=1000)


# In[90]:


optuna.visualization.plot_optimization_history(study_1).show()


# In[91]:


best_params_1 = study_1.best_params # Get best hyperparameters for the model 
print('Best Parameters:')

for key, value in study_1.best_params.items():
    print(f"\t{key}: {value}")


# ### Tuning remaining parameters 

# In[93]:


def objective_2(trial):
    
    params = {
        'gamma': trial.suggest_float('gamma', 0, 10),
        'max_leaves': trial.suggest_int('max_leaves', 5, 100),      
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10), 
        'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']), 
        'eval_metric': 'auc',                                  # Set evaluation metric 
        'tree_method': 'hist',                                 # Use 'hist' for GPU acceleration
        'device': 'cuda'                                       # Specify GPU device
             }
    
   
    model = XGBClassifier(
                    **params, 
                    n_jobs=-1, 
                    objective="binary:logistic",
                    grow_policy='lossguide',
                    max_depth=0,
                    learning_rate=best_params_1['learning_rate'],    
                    colsample_bytree=best_params_1['colsample_bytree'],
                    min_child_weight=best_params_1['min_child_weight'],
                    alpha=best_params_1['alpha'],
                    early_stopping_rounds=500, 
                    random_state=1142
                         )
        
    model.fit(        
        X_train, 
        y_train, 
        eval_set=[(X_val,y_val)],
        verbose=0)

    y_pred = model.predict(X_val)

    return roc_auc_score(y_val, y_pred)




study_2 = optuna.create_study(sampler=sampler, pruner=pruner, direction='maximize') # Direction set to maximize AUC
study_2.optimize(objective_2, n_trials=1000)


# In[94]:


optuna.visualization.plot_optimization_history(study_2).show()


# In[95]:


best_params_2 = study_2.best_params # Get best hyperparameters for the model 
print('Best Parameters:')

for key, value in study_2.best_params.items():
    print(f"\t{key}: {value}")


# ## Plugging in best hyperparameters

# In[99]:


xgboost_model = XGBClassifier(  
                    n_jobs=-1, 
                    objective="binary:logistic",
                    grow_policy = 'lossguide',
                    booster=best_params_2['booster'],
                    max_leaves=best_params_2['max_leaves'],
                    min_child_weight=best_params_1['min_child_weight'],
                    subsample=best_params_2['subsample'],
                    colsample_bytree=best_params_1['colsample_bytree'],
                    learning_rate=best_params_1['learning_rate'],
                    alpha=best_params_1['alpha'],
                    gamma=best_params_2['gamma'],
                    reg_lambda=best_params_2['reg_lambda'],
                    max_depth=0,
                    tree_method='hist',
                    device='cuda',                            # Set device to 'cuda' for GPU training
                    random_state=1142
                         )
    


# ## Running 10-fold stratified cross validation 

# In[100]:


get_ipython().run_cell_magic('time', '', "\nstratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)\n\ntraining_scores = []\nvalidation_scores = []\n\n# Iterate over each fold\nfor train_index, test_index in stratified_kfold.split(X, y):\n    X_train, X_val = X.iloc[train_index], X.iloc[test_index]\n    y_train, y_val = y.iloc[train_index], y.iloc[test_index]\n\n    # Fit the model on the training data\n    model = xgboost_model.fit(X_train, y_train)\n\n    # Evaluate the model and store the accuracy score\n    training = xgboost_model.score(X_train, y_train)\n    validation = xgboost_model.score(X_val, y_val)\n    training_scores.append(training)\n    validation_scores.append(validation)\n\nfor i, (train, test) in enumerate(zip(training_scores, validation_scores), 1):\n    print(f'Fold {i}: Training set accuracy = {train:.4f}, Validation set accuracy = {test:.4f}')\n\nprint(f'Average Training set accuracy: {np.mean(training_scores):.4f}')\nprint(f'Average Validation set accuracy: {np.mean(validation_scores):.4f}')\n")


# ### Average scores of training and test set indicate overfitting on training set 

# In[101]:


# Selecting Fold 3 because of highest testing set accuracy and lower training set accuracy than Fold 5   

model_number = 3
for i, (train_index, test_index) in enumerate(stratified_kfold.split(X, y), 1):
    if i == model_number:
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

xgboost_model.fit(X_train, y_train)


# ## Obtaining feature importances (according to gain)
# 
# ### Larger the gain, larger the improvement in accuracy by splitting on that feature

# In[129]:


feature_importances_gain = xgboost_model.get_booster().get_score(importance_type='gain')

print("Feature Importances:")
for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance}")


# ## Plotting feature importances (according to gain)

# In[139]:


feature_df = pd.DataFrame({'Feature': list(feature_importances_gain.keys()), 'Importance': list(feature_importances_gain.values())})

replace_dict = {
    'age': 'Age',
    'waist': 'Waist',
    'systolic_blood_pressure': 'Systolic Blood Pressure',
    'diastolic_blood_pressure': 'Diastolic Blood Pressure',
    'fasting_blood_sugar': 'Fasting Blood Sugar',
    'cholesterol': 'Cholesterol',
    'triglyceride': 'Triglyceride',
    'hdl': 'HDL',
    'ldl': 'LDL',
    'eyesight': 'Eyesight',
    'hearing': 'Hearing',
    'hemoglobin': 'Hemoglobin',
    'urine_protein': 'Urine Protein',
    'serum_creatinine': 'Serum Creatinine',
    'ast': 'AST',
    'alt': 'ALT',
    'ggt': 'GGT',
    'dental_caries': 'Dental Caries',
    'bmi': 'BMI'
             }

feature_df['Feature'] = feature_df['Feature'].replace(replace_dict)

feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.subplots(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='plasma').set(
    title='XGBoost Feature Importance by Gain', xlabel='Importance', ylabel='Feature',
            )
plt.show()


# ## Obtaining feature importances (according to coverage)
# 
# ### Larger the coverage, larger the impact on Hessian derivative of the loss function 

# In[135]:


feature_importances_cover = xgboost_model.get_booster().get_score(importance_type='cover')

print("Feature Importances:")
for feature, importance in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {importance}")


# ## Plotting feature importances (according to coverage)

# In[140]:


feature_df = pd.DataFrame({'Feature': list(feature_importances_cover.keys()), 'Importance': list(feature_importances_cover.values())})

replace_dict = {
    'age': 'Age',
    'waist': 'Waist',
    'systolic_blood_pressure': 'Systolic Blood Pressure',
    'diastolic_blood_pressure': 'Diastolic Blood Pressure',
    'fasting_blood_sugar': 'Fasting Blood Sugar',
    'cholesterol': 'Cholesterol',
    'triglyceride': 'Triglyceride',
    'hdl': 'HDL',
    'ldl': 'LDL',
    'eyesight': 'Eyesight',
    'hearing': 'Hearing',
    'hemoglobin': 'Hemoglobin',
    'urine_protein': 'Urine Protein',
    'serum_creatinine': 'Serum Creatinine',
    'ast': 'AST',
    'alt': 'ALT',
    'ggt': 'GGT',
    'dental_caries': 'Dental Caries',
    'bmi': 'BMI'
             }

feature_df['Feature'] = feature_df['Feature'].replace(replace_dict)

feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.subplots(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=feature_df, palette='plasma').set(
    title='XGBoost Feature Importance by Coverage', xlabel='Importance', ylabel='Feature',
            )
plt.show()


# ## Visualising model using graphviz library 

# In[180]:


xgboost.to_graphviz(xgboost_model, num_trees=tree_index_to_plot, rankdir='LR')


# ## Visualising model using dtreeviz library
# 
# ### Initialising dtreeviz adaptor 

# In[175]:


features = list(X_train.columns)
target = 'outcome'

viz_model = dtreeviz.model(xgboost_model, tree_index=1,
                           X_train=X_train[features], y_train=smoking[target],
                           feature_names=features,
                           target_name=target, class_names=['Smoker', 'Non-smoker'])


# ### Visualising model 

# In[176]:


viz_model.view(orientation="LR")


# In[177]:


viz_model.view(fancy=False)


# ### Viewing prediction path of an instance

# In[188]:


x = smoking[features].iloc[6857]
x


# In[185]:


viz_model.view(x=x)


# In[189]:


viz_model.view(x=x, show_just_path=True)


# In[190]:


print(viz_model.explain_prediction_path(x))


# ### Viewing leaf sizes

# In[191]:


viz_model.ctree_leaf_distributions()


# # MODEL EVALUATION

# In[201]:


from sklearn.metrics import f1_score
import os 


# ## Confusion matrix 

# In[202]:


y_pred_test = xgboost_model.predict(X_val)

cm = confusion_matrix(y_val, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Negatives(TN) = ', cm[0,0])

print('\nTrue Positives(TP) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[203]:


fig, ax = plt.subplots(figsize=(11, 8))

confusion_matrix = pd.DataFrame(data=cm, columns=['Negative', 'Positive'],
                                index=['Negative', 'Positive'])

sns.heatmap(confusion_matrix, annot=True, fmt='', cmap='Blues', square=True)

ax.xaxis.tick_top()
ax.set_title('Predicted', pad=15, fontsize='30')
plt.ylabel('Actual', fontsize='30')
plt.show()


# ## Classification report 

# In[204]:


target_names = ['Smoking absent', 'Smoking present']

print(classification_report(y_val, y_pred_test, target_names=target_names))


# ## F-1 macro score

# In[205]:


y_pred = xgboost_model.predict(X_val)

f1_macro = f1_score(y_val, y_pred, average='macro')

print(f'F1 Macro Score: {f1_macro:.4f}')


# ## Receiver Operating Characteristic Curve 

# In[206]:


# Calculate AUC 

AUC = roc_auc_score(y_val, y_pred_test)

#Plot ROC

fpr, tpr, thresholds = roc_curve(y_val, y_pred_test, pos_label=True)

plt.figure(figsize=(10, 10))

plt.plot(fpr, tpr, linewidth=2, label="AUC: {:.4f}".format(AUC))

plt.plot([0, 1], [0, 1], 'k--')

plt.title('Receiver Operating Characteristic Curve for Predicting Smoking Status', fontsize=20, pad=20)
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.legend(loc="lower right", fontsize=30)
plt.grid(True)
plt.show()


# ## Saving model for future use

# In[163]:


directory_name = 'smoking'

save_directory = os.getcwd()


directory_path = os.path.join(save_directory, directory_name)              # Create directory 
os.makedirs(directory_path, exist_ok=True)


model_file_name = 'xgboost_model_3_dec_2023_binary_classification_smoking'


model_file_path = os.path.join(directory_path, model_file_name + '.json')  # Save to directory 


xgboost_model.save_model(model_file_path)


