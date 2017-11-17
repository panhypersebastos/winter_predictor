# Last changes: dmasson 2017-10-05
# INFO:
# This code insert new (or even historical) monthly ERA-int data into MongoDB

# To run this code in a BATCH mode, enter the following command in the shell:
# python /home/dmasson/CloudStation/code/winter_predictor/era_interim_insert.py & 

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

num_cores = 3  # multiprocessing.cpu_count()

logfilename = '/home/dmasson/data/logfiles/era-interim_insert.log'
if os.path.exists(logfilename):
    os.remove(logfilename)
logging.basicConfig(filename=logfilename,
                    format='%(asctime)s %(message)s', level=logging.DEBUG)

startTime = datetime.now()
logging.info("%s %s:%s Job started" %
             (startTime.date(), startTime.hour, startTime.minute))
downloadDir = '/home/dmasson/data/era-interim/'
files00 = listdir(downloadDir)
files = fnmatch.filter(files00, '*multivarm1*.nc')
files.sort()

# What dates are already ingested in MongoDB ?
# MongoDB:
import sys
sys.path.insert(0, '/home/production/dev/')

mongo_host_local = 'mongodb://localhost:27017/'
mg = pymongo.MongoClient(mongo_host_local)

db = mg.ECMWF
con_data = db.ERAINT_monthly
datesInMongo = con_data.distinct('date')


def doIndexing():
    # Add indexes
    index1 = pymongo.IndexModel([("date", pymongo.DESCENDING)], name="date_-1")
    index2 = pymongo.IndexModel(
        [("id_grid", pymongo.ASCENDING), ("date", pymongo.DESCENDING)], name="id_grid_1_date_-1")
    index3 = pymongo.IndexModel(
        [("year", pymongo.ASCENDING), ("id_grid", pymongo.ASCENDING)], name="year_1_id_grid_1")
    mongo_host_local = 'mongodb://localhost:27017/'
    con = pymongo.MongoClient(mongo_host_local)
    db = con.ECMWF
    db.ERAINT_daily.create_indexes([index1, index2, index3])
    logging.info('--- Indexes added ---')


def insertToMongo(vars):
    # Stack all 2d arrays in one multi-d array
    this_day = vars['this_day']
    lons = vars['lons']
    lats = vars['lats']
    
    DAT = np.array([vars['ci'],   # 0
                    vars['sst'],  # 1
                    vars['istl1'],# 2
                    vars['sp'],   # 3
                    vars['stl1'], # 4
                    vars['msl'],  # 5
                    vars['u10'],  # 6
                    vars['v10'],  # 7
                    vars['t2m'],  # 8
                    vars['d2m'],  # 9
                    vars['al'],   # 10
                    vars['lcc'],  # 11
                    vars['mcc'],  # 12
                    vars['hcc'],  # 13
                    vars['si10'], # 14
                    vars['skt']   # 15
                    ])
    
    # Shift the grid so lons go from -180 to 180 instead of 0 to 360.
    DAT_shift, lons_shift = shiftgrid(
        lon0=180., datain=DAT, lonsin=lons, start=False)
    lon, lat = np.meshgrid(lons_shift, lats)
    this_dayhh = datetime.strptime(
        "%s-%s-%sT00:00:00Z" % (this_day.year, this_day.month, this_day.day), "%Y-%m-%dT%H:%M:%SZ")
    this_year = this_dayhh.year

    # Insert into MongoDB
    mongo_host_local = 'mongodb://localhost:27017/'
    con = pymongo.MongoClient(mongo_host_local)
    db = con.ECMWF

    #testlon = lon[51:53, 51:52]  # test with a smaller subset
    this_id = 0
    for (i, j), val in np.ndenumerate(lon):  # lon or testlon): !!!!!!!!!!
        this_id += 1
        db.ERAINT_monthly.insert_one({
            "id_grid": this_id,
            "date": this_dayhh,
            "year": this_year,
            "ci": round(DAT_shift[0, i, j], ndigits=2),
            "sst": round(DAT_shift[1, i, j], ndigits=2),
            "istl1": round(DAT_shift[2, i, j], ndigits=2),
            "sp": round(DAT_shift[3, i, j], ndigits=2),
            "stl1": round(DAT_shift[4, i, j], ndigits=2),
            "msl": round(DAT_shift[5, i, j], ndigits=2),
            "u10": round(DAT_shift[6, i, j], ndigits=2),
            "v10": round(DAT_shift[7, i, j], ndigits=2),
            "t2m": round(DAT_shift[8, i, j], ndigits=2),
            "d2m": round(DAT_shift[9, i, j], ndigits=2),
            "al": round(DAT_shift[10, i, j], ndigits=2),
            "lcc": round(DAT_shift[11, i, j], ndigits=2),
            "mcc": round(DAT_shift[12, i, j], ndigits=2),
            "hcc": round(DAT_shift[13, i, j], ndigits=2),
            "si10": round(DAT_shift[14, i, j], ndigits=2),
            "skt": round(DAT_shift[15, i, j], ndigits=2)
        })


def insertOneDay(this_day, ncfile, DF):
    # Choose one arbitrary day
    # this_day = days.iloc[0]
    logging.info(this_day)
    ncfile00 = '%s%s' % (downloadDir, ncfile)
    fh = Dataset(ncfile00, mode='r')
    lons = fh.variables['longitude'][:]
    lats = fh.variables['latitude'][:]
    # Extract the data for this day out of the nc file
    times = DF[DF.date == this_day].time
    ind = date2index(dates=times.tolist(), nctime=fh.variables['time'])

    vars = {'ci': fh.variables['ci'][ind], # Sea-ice cover [0-1]
            'sst': fh.variables['sst'][ind], # Sea surface temperature [K]
            'istl1': fh.variables['istl1'][ind], # Ice temp layer1 [K]
            'sp': fh.variables['sp'][ind], # Surface pressure [Pa]
            'stl1': fh.variables['stl1'][ind], # Soil temp lev1 [K]
            'msl': fh.variables['msl'][ind], # Mean SLP [Pa]
            'u10': fh.variables['u10'][ind], # wind-u [m/s]
            'v10': fh.variables['v10'][ind],
            't2m': fh.variables['t2m'][ind], # 2m temp [K]
            'd2m': fh.variables['d2m'][ind], # 2 metre dewpoint temperature[K]
            'al': fh.variables['al'][ind], # Surface albedo [0-1]
            'lcc': fh.variables['lcc'][ind], # Low cloud cover [0-1]
            'mcc': fh.variables['mcc'][ind], # Medium cloud cover [0-1]
            'hcc': fh.variables['hcc'][ind], # High cloud cover [0-1]
            'si10': fh.variables['si10'][ind], # 10m wind speed [m/s]
            'skt': fh.variables['skt'][ind], # Skin temperature [K]
            'lons': lons,
            'lats': lats,
            'this_day': this_day}

    insertToMongo(vars)
    if (this_day == date(1980, 1, 2)):
        # Setup the indexes just once
        doIndexing()

    fh.close()


def getDatesDF(nc_file):  # insertFile(nc_file):
    logging.info("Inserting %s" % (nc_file))
    nc_file00 = '%s%s' % (downloadDir, nc_file)
    fh = Dataset(nc_file00, mode='r')
    nctime = fh.variables['time'][:]
    t_unit = fh.variables['time'].units
    fh.close()
    time = num2date(nctime, units=t_unit)
    # Create a data frame
    df = pd.DataFrame({'time': time})
    df = df.assign(date=df.time.dt.date)
    # Do some aggregation
    gdf = pd.DataFrame(df.groupby('date').size().rename('ndoc')).reset_index()
    df2 = pd.merge(left=df, right=gdf, on="date")
    # exclude datesInMongo (data already ingested)
    DF = df2[~pd.to_datetime(df2.date).isin(datesInMongo)]
    return DF


for this_file in files:
    DF = getDatesDF(this_file)
    days = DF.date.drop_duplicates()
    # if (len(days) > 0):
    # filter(lambda x: insertOneDay(x, fh, df3), days) # 1cpu version
    # Parallel multicore version:

    def insertOneDayPar(d):
        insertOneDay(d, this_file, DF)
    Parallel(n_jobs=num_cores)(delayed(insertOneDayPar)(this_day)
                               for this_day in days)  # !!!!!!!!!!

do_checks = False
if do_checks is True:
    pprint(db.data.find_one())
    db.ERAINT_data.count()
    db.ERAINT_data.distinct(key='date')

endTime = datetime.now()
logging.info("%s %s:%s Job Done !!!" %
             (endTime.date(), endTime.hour, endTime.minute))
