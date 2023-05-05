import glob
import os
import pandas as pd
import seaborn as sns
import numpy as np
import mlflow

mlflow.autolog()

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings
warnings.filterwarnings('ignore')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

df = pd.read_csv('./outputs/datas/combined_dataset.csv')

df = df.drop(['track', 'artist', 'uri'], axis=1)

print(f'The dataset consists of {len(df)} rows.')

df.isnull().sum()

df.target.value_counts(normalize=True)

all_songs_hits = df.drop('target', axis=1).loc[df['target'] == 1]
all_songs_flops = df.drop('target', axis=1).loc[df['target'] == 0]

hits_means = pd.DataFrame(all_songs_hits.describe().loc['mean'])
flops_means = pd.DataFrame(all_songs_flops.describe().loc['mean'])
means_joined = pd.concat([hits_means,flops_means, (hits_means-flops_means)], axis = 1)
means_joined.columns = ['hit_mean', 'flop_mean', 'difference']
means_joined

means_joined.sort_values('difference', ascending=False)

pearson_corr = df.drop('target', axis=1).corr(method='pearson')
pearson_corr

spearman_corr = df.drop('target', axis=1).corr(method='spearman')
spearman_corr

df = df.drop(['duration_ms'], axis=1)

from sklearn.feature_selection import f_classif
anova_f_values = f_classif(df.drop(['target'], axis=1), df['target'])[0]

linear_corr = pd.Series(anova_f_values, index=df.drop(['target'], axis=1).columns)
linear_corr.sort_values(ascending=False)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)
print(len(df), len(df_full_train), len(df_train), len(df_val), len(df_test))

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

y_train = df_train.target.values
y_val = df_val.target.values
y_test = df_test.target.values


del df_train['target']
del df_val['target']
del df_test['target']

df_val

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

train_dicts = df_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

df_val.shape

X_val.shape


# XGB Classifier
import xgboost as xgb
from xgboost import XGBClassifier


features = dv.get_feature_names_out()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


xgb_params = {
    'eta': 0.3, 
    'max_depth': 60,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

watchlist = [(dtrain, 'train'), (dval, 'val')]

model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=5,
                  evals=watchlist)

y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)


y_pred = model.predict(dtrain)
roc_auc_score(y_train, y_pred)

# 
# The untuned XG Boost model has an accuracy of 85.92% on the validation dataset. 
# 
# Using grid search on XG Boost was very time-consuming so I am going to manually tune the following hyperparameters by plotting them and picking the best:
# 
# * eta
# * max_depth
# * min_child_weight

def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))
    
    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results

etas = {}

# % capture output

xgb_params = {
    'eta': 0.8, 
    'max_depth': 100,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}


model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=5,
                  evals=watchlist)

key = 'eta=%s' % (xgb_params['eta'])
# etas[key] = parse_xgb_output(output)
key

# From the plot, it is clear that eta=0.05 gives the best accuracy. 
scores = {}

xgb_params = {
    'eta': 0.05, 
    'max_depth': 40,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=5,
                  evals=watchlist)

key = 'max_depth=%s' % (xgb_params['max_depth'])
# scores[key] = parse_xgb_output(output)
key

# From the plot, it is clear that max_depth=40 gives the best accuracy. 

weights = {}

# %capture output

xgb_params = {
    'eta': 0.05, 
    'max_depth': 40,
    'min_child_weight': 10,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=5,
                  evals=watchlist)


key = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
# weights[key] = parse_xgb_output(output)
key

# From the plot, it is clear the min_child_weight=1 gives the best accuracy on validation dataset.

xgb_params = {
    'eta': 0.05, 
    'max_depth': 40,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=20,
                  evals=watchlist)

y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

# Tuning the hyperparameters manually boosted the accuracy from 85.92% to 86.37% on the validation dataset.

# ## Selecting the best model
# ### XG Boost
# 

xgb_params = {
    'eta': 0.05, 
    'max_depth': 40,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200) # ,evals=watchlist)

y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

# The best performing model is XG Boost with an accuracy of 88.36% on the validation dataset, it will be exported to a Pickle file to be used with the Flask application. 

# ## Store best model as Pickle file

df_full_train = df_full_train.reset_index(drop=True)
y_full_train = df_full_train.target.values
del df_full_train['target']

dv = DictVectorizer(sparse=False)

dicts_full_train = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                    feature_names=dv.get_feature_names_out())

dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out())

xgb_params = {
    'eta': 0.05, 
    'max_depth': 40,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=200)

y_pred = model.predict(dtest)
roc_auc_score(y_test, y_pred)

# The accuracy of the full_train dataset is 85.97% which is close to the validation dataset accuracy. 
# The model will be exported to a Pickle file to be used with the Flask application.
import pickle

output_file = './outputs/models/final_model_xgb.bin'

print("enregistrement du modele dans le fichier: ", output_file)

f_out = open(output_file, 'wb') 
pickle.dump((dv, model), f_out)
f_out.close()

with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model), f_out)

input_file = './outputs/models/final_model_xgb.bin'

with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)

song = df_full_train.to_dict(orient='records')[0]

X = dv.transform([song])
dX = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())

y_pred = model.predict(dX)

print('input:', song)
print('output:', y_pred)