# Last changes: dmasson 2017-10-05
# INFO:
# This code does the Mongodb indexing of monthly ERA-int data if
# not done before

# To run this code in a BATCH mode, enter the following command in the shell:
# python /home/dmasson/CloudStation/code/winter_predictor/era_interim_indexing.py & 

from netCDF4 import Dataset, netcdftime, num2date, date2num, date2index
from datetime import datetime, timedelta, date
import pytz
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import pymongo
from pymongo import IndexModel, ASCENDING, DESCENDING
from pprint import pprint
from os import listdir
import os
import pandas as pd
import fnmatch
import logging
from joblib import Parallel, delayed
import multiprocessing
from functools import partial

ERA_vers = 'lores' # or 'hires'

if (ERA_vers == 'hires'):
    col_dat = 'ERAINT_monthly'
    col_grid = 'ERAINT_grid'
    resolution = 0.25
elif (ERA_vers == 'lores'):
    col_dat = 'ERAINT_lores_monthly'
    col_grid = 'ERAINT_lores_grid'
    resolution = 2.5

logfilename = '/home/dmasson/data/logfiles/era-interim_indexing.log'
if os.path.exists(logfilename):
    os.remove(logfilename)
logging.basicConfig(filename=logfilename,
                    format='%(asctime)s %(message)s', level=logging.DEBUG)

startTime = datetime.now()
logging.info("%s %s:%s Job started" %
             (startTime.date(), startTime.hour, startTime.minute))

mongo_host_local = 'mongodb://localhost:27017/'
mg = pymongo.MongoClient(mongo_host_local)

db = mg.ECMWF
con_data = db[col_dat]

def doIndexing():
    # Add indexes
    logging.info('--- Starting indexing ---')
    index1 = pymongo.IndexModel([("date", pymongo.DESCENDING)], name="date_-1")
    index2 = pymongo.IndexModel([("id_grid", pymongo.ASCENDING), ("date", pymongo.DESCENDING)], name="id_grid_1_date_-1")
    index3 = pymongo.IndexModel([("year", pymongo.ASCENDING), ("id_grid", pymongo.ASCENDING)], name="year_1_id_grid_1")
    mongo_host_local = 'mongodb://localhost:27017/'
    con = pymongo.MongoClient(mongo_host_local)
    db = con.ECMWF
    db[col_dat].create_indexes([index1, index2, index3])
    logging.info('--- Indexes added ---')

doIndexing()
