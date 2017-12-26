# This code create a new data collection in Mongo DB.
# The new collection contains the anomalies of the original
# ERA-interim time series for many variables.
# As the anomalies are the resiuals resulting from
# a linear model, the code does as detrending and
# a de-seasonalization at the same time.

# In order to run the script as a BATCH job, execute:
# python ~/CloudStation/code/winter_predictor/detrending_new_collection.py & 

import numpy as np
import matplotlib.pyplot as plt
import pymongo
from pprint import pprint
from datetime import datetime, timedelta, date
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import sklearn.linear_model as skl_lm
from itertools import compress

mongo_host_local = 'mongodb://localhost:27017/'
mg = pymongo.MongoClient(mongo_host_local)
db = mg.ECMWF

ERA_vers = 'lores'
if (ERA_vers == 'hires'):
    col_dat = 'ERAINT_monthly'
    col_grid = 'ERAINT_grid'
    resolution = 0.25
elif (ERA_vers == 'lores'):
    col_dat = 'ERAINT_lores_monthly'
    col_grid = 'ERAINT_lores_grid'
    resolution = 2.5

con_data = db[col_dat]

# Get the name of all physical variables
fo = con_data.find_one()
keynames = pd.Series(list(fo.keys()))
vind = keynames.isin(['_id', 'id_grid', 'date', 'year'])
varnames = list(compress(keynames, ~vind))


this_id_grid = 777
this_month = 1
this_variable = 'z70'

qry = {"id_grid": this_id_grid}
res = con_data.find(filter=qry, projection={"_id": 0})
df = pd.DataFrame(list(res))

# Create a new column with the month index
df = df.assign(month=list(map(lambda x: x.month, df.date)))
df.head()

# Atomic detrending
# * Extract anomaly with regard to long-term trend
# * Because the de-trending is done for each month,
#   it also serves as de-seasonaization

ts = df.query('month == %s' % (this_month)).\
     sort_values('date').\
     reset_index(drop=True)
# Covariate is the number of days since the beg. of TS
X = (ts.date - ts.date[0]).dt.days.values.reshape(-1, 1)


def getAnom(vn):
    model = skl_lm.LinearRegression()
    y = ts[[vn]]
    model.fit(X, y)
    lm_pred = model.predict(X)
    resid = lm_pred - y
    return resid

anom_df = pd.concat(list(map(getAnom, varnames)), axis=1)
anom_df.shape
anom_df.z70
