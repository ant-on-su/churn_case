# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import warnings
import os

import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from ydata_profiling import ProfileReport
import sweetviz as sw
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
# -

# ## 1. Read Data

filename = "./data/assesment_file2_churn.csv"
df_source = pd.read_csv(filename, sep=";")
df_source

# ## 2. Check the Data and the columns

# +
# NA in columns

df_source.isna().sum()

# +
# Data types

df_source.dtypes

# +
# Check columns: "TARGET" and "Record_Count", if it is not redundant

df_source["TARGET"].unique()
# -

df_source["Record_Count"].unique()

# OK, those columns can be safely dropped
#
# **What values do we have in the 'object'-type cols?**

df_source["CLIENTGROUP"].unique()

df_source["ACCOUNTMODEL"].unique()

list(df_source["AGE_CLASS"].unique())

df_source["HOMEBANK_COLOUR"].unique()

df_source["LOYALITY"].unique()

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ## 3. Dive in a few indivudual cases

# +
# 1 Customer churned at start:

df_source[df_source["CUSTOMER_ID"]==12284116].sort_values(by="MONTH_PERIOD")

# +
# 2 Customer churned half-way:

df_source[df_source["CUSTOMER_ID"]==42668312].sort_values(by="MONTH_PERIOD")
# Notice: when churned, obj-cols become NA, also first CHURNED_IND==1, only after COMMERCIALLY_CHURNED==1

# +
# 3 Another customer churned half-way:

df_source[df_source["CUSTOMER_ID"]==81085578].sort_values(by="MONTH_PERIOD")
# Notice: COMMERCIALLY_CHURNED==1 now before CHURNED_IND==1

# +
# 4 Yet another customer churned half-way:

df_source[df_source["CUSTOMER_ID"]==9986098607].sort_values(by="MONTH_PERIOD")
# Notice: Leeftijd_onbekend and Colours become NA before both churn indicators (possibly died, but not officially churned yet?)

# +
# 5 Customer churned at the end:

df_source[df_source["CUSTOMER_ID"]==4106074447].sort_values(by="MONTH_PERIOD")
# Notice: Same, Leeftijd_onbekend and Colours become NA before both churn indicators
# -

# ## 4. Data manipulations
# Let's clean a data just a bit and add some useful features

# +
# Let's work on a copy to not mess up the original
df = df_source.copy(deep=True)

# As either COMMERCIALLY_CHURNED and CHURNED_IND means actual churn-event, and they seem not ecrearly connected, let's summarise them in one col:
df["churn"] = df[["CHURNED_IND","COMMERCIALLY_CHURNED"]].max(axis=1)

# Add col with earliest churn month timestamp:
df["churn_month"] = df.groupby("CUSTOMER_ID")["MONTH_PERIOD"].transform(lambda x: x.where(df['churn'] == 1).min())

# And drop all the records beyond the churn event:
df = df[(df["MONTH_PERIOD"] < df["churn_month"]) | (df["churn_month"].isna())]

# Add col "duration" showing number of months between the record motnth and the churn month:
df["duration"] = (df["churn_month"]//100 - df["MONTH_PERIOD"]//100)*12 + (df["churn_month"]%100 - df["MONTH_PERIOD"]%100)

# Add col "churned" as per customer indicator whether a customer has eventually churned:
df["churned"] = df["duration"].notna().astype(int)

# Now we can drop unnecessary columns:
df.drop(columns=["TARGET","Record_Count","CHURNED_IND","COMMERCIALLY_CHURNED","churn"], inplace=True)

# And see what we've got
df
# -

# ### Let's get some stats regarding our targets, per customer

print(f'Total customers:\t{df.groupby("CUSTOMER_ID")["churned"].count().count()}')
print(f'Churned customers:\t{df.groupby("CUSTOMER_ID")["churned"].max().sum()}')
print(f'% churned customers:\t{df.groupby("CUSTOMER_ID")["churned"].max().mean()}')
print(f'% churned records:\t{df.churned.mean()}')

# ### Check how the customers churning over time looks like

# Duration in months per customer
sns.histplot(df.groupby("CUSTOMER_ID")["duration"].max(), bins=23)

for mon in range(2,24):
    print(f'{mon} months: {(df.groupby("CUSTOMER_ID")["duration"].max() >=mon).sum()}')

# ## 5. Data exploration

# + jupyter={"outputs_hidden": true}
import hiplot as hip
df_hiplot = hip.Experiment.from_dataframe(df.drop(columns=["CUSTOMER_ID","MONTH_PERIOD"]))
df_hiplot.display()

# +
reports_path = "./reports/"

print(f"Generating Pandas profile report")
ydata_rep = ProfileReport(df, title="Churn EDA")
ydata_rep.to_file(f"{reports_path}pandas_report.html")

print(f"Generating Sweetviz report")
sw_rep = sw.analyze(df)
sw_rep.show_html(f"{reports_path}sw_report.html")
# -

# ## 6. Data preprocessing

# +
data_df = df.copy(deep=True)

data_df.drop(columns="churn_month",inplace=True)

# +
# Encode categorical features

data_df["AGE_CLASS"] = data_df["AGE_CLASS"].map({'Leeftijd_12_17   ':1,
                                                 'Leeftijd_18_23   ':2,
                                                 'Leeftijd_24_29   ':3,
                                                 'Leeftijd_30_34   ':4,
                                                 'Leeftijd_35_40   ':5,
                                                 'Leeftijd_41_54   ':6,
                                                 'Leeftijd_55_64   ':7,
                                                 'Leeftijd_65_74   ':8,
                                                 'Leeftijd_75_plus ':9,
                                                 'Leeftijd_onbekend':0,
                                                })

data_df["LOYALITY"] = data_df["LOYALITY"].map({
                                               'Groen   ':0,
                                               'Wit     ':1,
                                               'Oranje  ':2,
                                               'Rood    ':3,
    
})
data_df["HOMEBANK_COLOUR"] = data_df["HOMEBANK_COLOUR"].map({'Groen   ':0,
                                                             'Geel    ':1,
                                                             'Oranje  ':2,
                                                             'Rood    ':3,
    
})


data_df["ACCOUNTMODEL"] = data_df["ACCOUNTMODEL"].map({'LP':0,
                                                       'MP':1,
                                                       'HW':2,
                                                       'HP':3,
                                                       
    
})

data_df = pd.get_dummies(data_df, columns=["CLIENTGROUP"], prefix=["CLIENTGROUP="] )
# data_df = data_df.drop(columns=["CLIENTGROUP"])

# data_df.loc[data_df.churned==0, "duration"] = (date_max//100 - df["MONTH_PERIOD"]//100)*12 + (date_max%100 - df["MONTH_PERIOD"]%100)

data_df = data_df.fillna(-1)
# -

data_df

# ## 7. BASELINE Model
#
# The idea is to quickly check if we can distinguish churn generally, say, taking the means of all features over the whole period

# +
# Create the df per customer with the means of all features
# we are going to use "churned" as boolean target

baseline_df = data_df.drop(columns=["MONTH_PERIOD","duration"]).groupby("CUSTOMER_ID").mean()
baseline_df
# -

# How many churned customers we have in total?
baseline_df["churned"].sum()

# +
# Train test split

X = baseline_df.drop(columns="churned")
y = baseline_df["churned"]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)


# -

def score_model(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_train: pd.Series,
    y_pred_test: pd.Series,
) -> dict[str, float]:
    """
    Score model on the multiple metrics and return a dictionary
    """

    return {
        "accuracy_train": accuracy_score(y_train, y_pred_train),
        "accuracy_test": accuracy_score(y_test, y_pred_test),
        "f1_train": f1_score(y_train, y_pred_train),
        "f1_test": f1_score(y_test, y_pred_test),
        "recall_train": recall_score(y_train, y_pred_train),
        "recall_test": recall_score(y_test, y_pred_test),
        "precision_train": precision_score(y_train, y_pred_train),
        "precision_test": precision_score(y_test, y_pred_test),
        "roc_auc_score_train": roc_auc_score(y_train, y_pred_train),
        "roc_auc_score_test": roc_auc_score(y_test, y_pred_test),
    }


# +
# Let's try several classifiers to get the idea of what can work better:

random_state = 123

clf_dict={
    "Logistic Regression" : LogisticRegression(random_state = random_state),
    "LinearSVC" : LinearSVC(random_state = random_state),
    "Desision Tree" : DecisionTreeClassifier(random_state = random_state),
    "Random_forest" : RandomForestClassifier(random_state = random_state),
    
}

score_clf_dict = {}

for clf_name,clf in clf_dict.items():
    
    pipe = make_pipeline(StandardScaler(), clf)
    pipe.fit(X_train, y_train)

    y_pred_test = pipe.predict(X_test)
    y_pred_train = pipe.predict(X_train)
    
    print(f"\nUsing {clf_name}:")
    print(classification_report(y_test,y_pred_test))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test,y_pred_test))
    
    score_clf_dict[clf_name] = score_model(y_train, y_test, y_pred_train, y_pred_test)

# -

# ## 8. Next Model

stride =1
size =3
start = 1
num_mnt_back = 2
gap = 6

# +
ts_unc_data_df = pd.DataFrame()
ts_cen_data_df = pd.DataFrame()

for end_mnt in range(start,num_mnt_back+start-1, stride):
    start_mnt = end_mnt+size-1
    select_df = data_df[(data_df["churned"]==1)&(data_df["duration"]<=start_mnt)&(data_df["duration"]>=end_mnt)]

    mean_feat_df = select_df.groupby("CUSTOMER_ID").mean()
    mean_feat_df.drop(columns=["MONTH_PERIOD","duration","churned"],inplace = True)
    mean_feat_df.columns = [f"{col}_mean" for col in mean_feat_df.columns]

    start_feat_df = select_df[data_df["duration"]==start_mnt].set_index("CUSTOMER_ID")
    start_feat_df.drop(columns=["MONTH_PERIOD","duration","churned"],inplace = True)

    end_feat_df = select_df[data_df["duration"]==end_mnt].set_index("CUSTOMER_ID")
    end_feat_df.drop(columns=["MONTH_PERIOD","duration","churned"],inplace = True)
    
    diff_df = end_feat_df - start_feat_df
    diff_df.columns = [f"{col}_diff" for col in diff_df.columns]

    window_df = mean_feat_df.merge(diff_df,left_index=True, right_index=True)
    window_df["duration"] = end_mnt
    window_df["churned"] = 1
    window_df.reset_index(inplace=True)
    ts_unc_data_df = pd.concat([ts_unc_data_df,window_df],ignore_index=True).dropna()
    print(f"unchurned: window {end_mnt} resulted in {ts_unc_data_df.shape[0]} rows")
    
for end_mnt in range(start,num_mnt_back+start-1, stride):
    start_mnt = end_mnt+size-1
    end_mp = data_df["MONTH_PERIOD"].max()%100 - end_mnt%12 + (data_df["MONTH_PERIOD"].max()//100 - end_mnt//12)*100
    start_mp = data_df["MONTH_PERIOD"].max()%100 - start_mnt%12 + (data_df["MONTH_PERIOD"].max()//100 - start_mnt//12)*100
    
    start_mnt = end_mnt+size-1
    select_df = data_df[(data_df["churned"]==0)&(data_df["MONTH_PERIOD"]>=start_mp)&(data_df["MONTH_PERIOD"]<=end_mp)]

    mean_feat_df = select_df.groupby("CUSTOMER_ID").mean()
    mean_feat_df.drop(columns=["MONTH_PERIOD","duration","churned"],inplace = True)
    mean_feat_df.columns = [f"{col}_mean" for col in mean_feat_df.columns]

    start_feat_df = select_df[data_df["MONTH_PERIOD"]==start_mp].set_index("CUSTOMER_ID")
    start_feat_df.drop(columns=["MONTH_PERIOD","duration","churned"],inplace = True)

    end_feat_df = select_df[data_df["MONTH_PERIOD"]==end_mp].set_index("CUSTOMER_ID")
    end_feat_df.drop(columns=["MONTH_PERIOD","duration","churned"],inplace = True)

    diff_df = end_feat_df - start_feat_df
    diff_df.columns = [f"{col}_diff" for col in diff_df.columns]

    window_df = mean_feat_df.merge(diff_df,left_index=True, right_index=True)
    window_df["duration"] = end_mnt + gap
    window_df["churned"] = 0
    window_df.reset_index(inplace=True)
    ts_cen_data_df = pd.concat([ts_cen_data_df,window_df],ignore_index=True).dropna()
    print(f"churned: window {end_mnt} resulted in {ts_cen_data_df.shape[0]} rows")
    
    
ts_data_df = pd.concat([ts_unc_data_df,ts_cen_data_df],ignore_index=True).dropna()
# -

ts_data_df

# +
num_customers = ts_data_df.groupby("CUSTOMER_ID")["churned"].max()

print(f"Customers churned:\t{num_customers.sum()}")
print(f"Customers not churned:\t{num_customers.count()-num_customers.sum()}")


# -

def ts_train_test_split(df, test_size=0.2, stratify=None, random_state=None):
    
    idx_df = pd.concat([df.groupby("CUSTOMER_ID")["duration"].min(),df.groupby("CUSTOMER_ID")["churned"].min()], axis=1)
    
    if stratify:
        stratify = idx_df[stratify]
    idx_train, idx_test = train_test_split(idx_df, test_size=test_size, stratify=stratify, random_state=random_state)
    
    XY_train = df[df["CUSTOMER_ID"].isin(idx_train.index)]
    XY_test = df[df["CUSTOMER_ID"].isin(idx_test.index)]
    
    X_train = XY_train.set_index(["CUSTOMER_ID","duration"]).drop(columns=["churned"])
    Y_train = XY_train.set_index(["CUSTOMER_ID","duration"],drop=False)[["churned","duration"]]
    
    X_test = XY_test.set_index(["CUSTOMER_ID","duration"]).drop(columns=["churned"])
    Y_test = XY_test.set_index(["CUSTOMER_ID","duration"],drop=False)[["churned","duration"]]
    
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = ts_train_test_split(ts_data_df, test_size=0.2, stratify="churned", random_state=123)

# +
regr = RandomForestRegressor(random_state=123)

regr.fit(X_train, Y_train["duration"])
score = regr.score(X_test, Y_test["duration"])
    
print(score)


# +
clf = RandomForestClassifier(random_state=123)

clf.fit(X_train, Y_train["churned"])

y_pred_test = clf.predict(X_test)

print("Using RandomForestClassifier:")
print(classification_report(Y_test["churned"],y_pred_test))
print("Confusion Matrix:")
print(confusion_matrix(Y_test["churned"],y_pred_test))
# -

result_df = Y_test
result_df["pred"] = y_pred_test
result_df


