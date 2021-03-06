# This code create a new data collection in Mongo DB.
# The new collection contains the anomalies of the original
# ERA-interim time series for many variables.
# As the anomalies are the resiuals resulting from
# a linear model, the code does as detrending and
# a de-seasonalization at the same time.

# In order to run the script as a BATCH job, execute:
# python ~/CloudStation/code/winter_predictor/detrending_new_collection.py & 

import numpy as np
import pymongo
import pandas as pd
import sklearn.linear_model as skl_lm
from itertools import compress
from joblib import Parallel, delayed
import os
import logging
from datetime import datetime

num_cores = 3  # multiprocessing.cpu_count()

logfilename = '/home/dmasson/data/logfiles/detrending_new_collection.log'
if os.path.exists(logfilename):
    os.remove(logfilename)
logging.basicConfig(filename=logfilename,
                    format='%(asctime)s %(message)s', level=logging.DEBUG)

startTime = datetime.now()
logging.info("%s %s:%s Job started" %
             (startTime.date(), startTime.hour, startTime.minute))


mongo_host_local = 'mongodb://localhost:27017/'
mg = pymongo.MongoClient(mongo_host_local, connect=False)
db = mg.ECMWF

ERA_vers = 'lores'
if (ERA_vers == 'hires'):
    col_dat = 'ERAINT_monthly'
    col_anom = 'ERAINT_monthly_anom'
    col_grid = 'ERAINT_grid'
    resolution = 0.25
elif (ERA_vers == 'lores'):
    col_dat = 'ERAINT_lores_monthly'
    col_anom = 'ERAINT_lores_monthly_anom'
    col_grid = 'ERAINT_lores_grid'
    resolution = 2.5

con_data = db[col_dat]

# Get the name of all physical variables
fo = con_data.find_one()
keynames = pd.Series(list(fo.keys()))
vind = keynames.isin(['_id', 'id_grid', 'date', 'year'])
varnames = list(compress(keynames, ~vind))


def insertToMongo(this_id_grid, this_month):
    mg = pymongo.MongoClient(mongo_host_local, connect=False)
    db = mg.ECMWF
    con_data = db[col_dat]
    qry = {"id_grid": this_id_grid}
    res = con_data.find(filter=qry, projection={"_id": 0})
    df = pd.DataFrame(list(res))

    # Create a new column with the month index
    df = df.assign(month=list(map(lambda x: x.month, df.date)))

    # Atomic detrending
    # * Extract anomaly with regard to long-term trend
    # * Because the de-trending is done for each month,
    #   it also serves as de-seasonaization

    ts = df.query('month == %s' % (this_month)).\
        sort_values('date').\
        reset_index(drop=True)
    # Covariate is the number of days since the beg. of TS
    X = (ts.date - ts.date[0]).dt.days.values.reshape(-1, 1)

    def getAnom(vn):  # 'vn' is the variable name
        model = skl_lm.LinearRegression()
        y = ts[[vn]]
        model.fit(X, y)
        lm_pred = model.predict(X)
        resid = lm_pred - y
        return resid

    anom_df = pd.concat(list(map(getAnom, varnames)), axis=1)
    anom_df = pd.concat([ts[['date', 'year', 'month']], anom_df], axis=1)
    anom_df = anom_df.assign(id_grid=this_id_grid)

    # Insert this dataframe in MongoDb
    con_anom = db[col_anom]
    anom_dict = anom_df.to_dict(orient='records')
    con_anom.insert_many(anom_dict)


grid_list = db[col_grid].distinct(key='id_grid')
months_list = np.arange(1, 12+1)

# Parallel insertion
Parallel(n_jobs=num_cores)(delayed(insertToMongo)(i, j)
                           for i in grid_list for j in months_list)

endTime = datetime.now()
logging.info("%s %s:%s Job Done !!!" %
             (endTime.date(), endTime.hour, endTime.minute))

# Create Index by executing 'era_interim_indexing.py' on the
# proper collection
