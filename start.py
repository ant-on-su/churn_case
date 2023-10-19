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
import pandas as pd
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
# -

filename = "./data/assesment_file2_churn.csv"
df = pd.read_csv(filename, sep=";")
df

df.isna().sum()

df.dtypes

df["TARGET"].unique()

df["Record_Count"].unique()

profile = ProfileReport(df, title="Churn EDA")

profile.to_file("./reports/report_ydata.html")

profile_ts = ProfileReport(df, tsmode=True, sortby="MONTH_PERIOD",title="Churn EDA")
profile_ts.to_file("./reports/report_ts_ydata.html")

df.groupby("CUSTOMER_ID").count().min()

df.groupby("MONTH_PERIOD")["CHURNED_IND","COMMERCIALLY_CHURNED"].sum()

client_churn_df = df.groupby("CUSTOMER_ID")["CHURNED_IND","COMMERCIALLY_CHURNED"].max()
client_churn_df 

client_churn_df.sum()

client_rep_churn_df = df.groupby("CUSTOMER_ID")["CHURNED_IND","COMMERCIALLY_CHURNED"].sum()
client_rep_churn_df

client_rep_churn_df.max()

client_rep_churn_df[client_rep_churn_df["CHURNED_IND"]>0].sort_values(by="CHURNED_IND")

df[df["CUSTOMER_ID"]==12284116].sort_values(by="MONTH_PERIOD")

df[df["CUSTOMER_ID"]==42668312].sort_values(by="MONTH_PERIOD")

df[df["CUSTOMER_ID"]==81085578].sort_values(by="MONTH_PERIOD")

df[df["CUSTOMER_ID"]==9986098607].sort_values(by="MONTH_PERIOD")

df[df["CUSTOMER_ID"]==6268711221].sort_values(by="MONTH_PERIOD")

df[df["CUSTOMER_ID"]==4106074447].sort_values(by="MONTH_PERIOD")


