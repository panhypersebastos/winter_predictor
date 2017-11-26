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
ERA_vers = 'lores' # or 'hires'

if (ERA_vers == 'hires'):
    col_dat = 'ERAINT_monthly'
    col_grid = 'ERAINT_grid'
    resolution = 0.25
elif (ERA_vers == 'lores'):
    col_dat = 'ERAINT_lores_monthly'
    col_grid = 'ERAINT_lores_grid'
    resolution = 2.5

downloadDir = '/home/dmasson/data/era-interim/%s/' % (ERA_vers)

logfilename = '/home/dmasson/data/logfiles/era-interim_insert.log'
if os.path.exists(logfilename):
    os.remove(logfilename)
logging.basicConfig(filename=logfilename,
                    format='%(asctime)s %(message)s', level=logging.DEBUG)

startTime = datetime.now()
logging.info("%s %s:%s Job started" %
             (startTime.date(), startTime.hour, startTime.minute))

files00 = listdir(downloadDir)
files = fnmatch.filter(files00, 'era-int_file01*.nc')
files.sort()

prefixes = list(map(lambda x: 'era-int_file%s_%s_' % (x,resolution),
                    ['01', '02', '03']))
postfixes = list(map(lambda x: x[19:], files))

files_df = pd.DataFrame(
    {'period': postfixes,
     'f1': list(map(lambda x: '%s%s' % (prefixes[0], x), postfixes)),
     'f2': list(map(lambda x: '%s%s' % (prefixes[1], x), postfixes)),
     'f3': list(map(lambda x: '%s%s' % (prefixes[2], x), postfixes))})


# What dates are already ingested in MongoDB ?
# MongoDB:
import sys
sys.path.insert(0, '/home/production/dev/')

mongo_host_local = 'mongodb://localhost:27017/'
mg = pymongo.MongoClient(mongo_host_local)

db = mg.ECMWF
con_data = db[col_dat]
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
    db[col_dat].create_indexes([index1, index2, index3])
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
                    vars['skt'],  # 15
                    vars['blh'],  # 16
                    vars['ishf'], # 17
                    vars['ie'],   # 18
                    vars['z70']   # 19
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
        db[col_dat].insert_one({
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
            "skt": round(DAT_shift[15, i, j], ndigits=2),
            "blh": round(DAT_shift[16, i, j], ndigits=2),
            "ishf": round(DAT_shift[17, i, j], ndigits=2),
            "ie": round(DAT_shift[18, i, j], ndigits=2),
            "z70": round(DAT_shift[19, i, j], ndigits=2)
        })


def insertOneDay(this_day, ncfiles, DF):
    # Choose one arbitrary day
    # this_day = days.iloc[0]
    logging.info(this_day)
    ncfile01 = '%s%s' % (downloadDir, ncfiles[0])
    ncfile02 = '%s%s' % (downloadDir, ncfiles[1])
    ncfile03 = '%s%s' % (downloadDir, ncfiles[2])
    fh01 = Dataset(ncfile01, mode='r')
    fh02 = Dataset(ncfile02, mode='r')
    fh03 = Dataset(ncfile03, mode='r')
    lons = fh01.variables['longitude'][:]
    lats = fh01.variables['latitude'][:]
    # Extract the data for this day out of the nc file
    times = DF[DF.date == this_day].time
    ind01 = date2index(dates=times.tolist(), nctime=fh01.variables['time'])
    ind02 = date2index(dates=times.tolist(), nctime=fh02.variables['time'])
    ind03 = date2index(dates=times.tolist(), nctime=fh03.variables['time'])

    vars = {'ci': fh01.variables['ci'][ind01], # Sea-ice cover [0-1]
            'sst': fh01.variables['sst'][ind01], # Sea surface temperature [K]
            'istl1': fh01.variables['istl1'][ind01], # Ice temp layer1 [K]
            'sp': fh01.variables['sp'][ind01], # Surface pressure [Pa]
            'stl1': fh01.variables['stl1'][ind01], # Soil temp lev1 [K]
            'msl': fh01.variables['msl'][ind01], # Mean SLP [Pa]
            'u10': fh01.variables['u10'][ind01], # wind-u [m/s]
            'v10': fh01.variables['v10'][ind01],
            't2m': fh01.variables['t2m'][ind01], # 2m temp [K]
            'd2m': fh01.variables['d2m'][ind01], # 2m dewpoint temp.[K]
            'al': fh01.variables['al'][ind01], # Surface albedo [0-1]
            'lcc': fh01.variables['lcc'][ind01], # Low cloud cover [0-1]
            'mcc': fh01.variables['mcc'][ind01], # Medium cloud cover [0-1]
            'hcc': fh01.variables['hcc'][ind01], # High cloud cover [0-1]
            'si10': fh01.variables['si10'][ind01], # 10m wind speed [m/s]
            'skt': fh01.variables['skt'][ind01], # Skin temperature [K]
            'blh': fh02.variables['blh'][ind02], # Boundary layer hgt [m]
            'ishf': fh02.variables['ishf'][ind02],# Inst.surf.sensbl.heatflux [W/m2]
            'ie': fh02.variables['ie'][ind02],# Instantaneous moisture flux [kg*m^-2*s^-1]
            'z70': fh03.variables['z'][ind03], # Geopot. height @70hPa [m]
            'lons': lons,
            'lats': lats,
            'this_day': this_day}

    insertToMongo(vars)
    if (this_day == date(1980, 1, 1)):
        # Setup the indexes just once
        doIndexing()

    fh01.close()
    fh02.close()
    fh03.close()


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


for this_period in files_df.period:
    these_files = [files_df.query('period==@this_period').f1.item(),
                   files_df.query('period==@this_period').f2.item(),
                   files_df.query('period==@this_period').f3.item()]
    DF = getDatesDF(these_files[0])
    days = DF.date.drop_duplicates()
    # if (len(days) > 0):
    # filter(lambda x: insertOneDay(x, fh, df3), days) # 1cpu version
    # Parallel multicore version:

    def insertOneDayPar(d):
        insertOneDay(d, these_files, DF)
    Parallel(n_jobs=num_cores)(delayed(insertOneDayPar)(this_day)
                               for this_day in days)

do_checks = False
if do_checks is True:
    pprint(db.data.find_one())
    db[col_dat].count()
    db[col_dat].distinct(key='date')

endTime = datetime.now()
logging.info("%s %s:%s Job Done !!!" %
             (endTime.date(), endTime.hour, endTime.minute))
