# This code downloads newest ERA data
# Last modification: massond 2017-11-12

# To run this code in a BATCH mode, enter the following command in the shell:
# python ~/CloudStation/code/winter_predictor/era_interim_download_monthly.py > ~/data/logfiles/era_interim_download_monthly.log & 

from dateutil.relativedelta import relativedelta
from ecmwfapi import ECMWFDataServer
import logging
import pymongo
from os import listdir
import os
import fnmatch

from datetime import datetime, date
import numpy as np
import itertools
import pandas as pd

from era_interim_download_functions import getMultivarMon

downloadDir = '/home/dmasson/data/era-interim'

logfilename = '/home/dmasson/data/logfiles/era-interim_download.plog'
if os.path.exists(logfilename):
    os.remove(logfilename)
logging.basicConfig(filename=logfilename,
                    format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.info('Job started')

# Query the latest data in MongoDB
mongo_host_local = 'mongodb://localhost:27017/'
mg = pymongo.MongoClient(mongo_host_local)
db = mg.ECMWF
con_data = db.ERAINT_monthly
datesInMongo = con_data.distinct('date')

try:
    lastIngested = max(datesInMongo)
except:
    print('No data in mongodb, assigning to default')
    lastIngested = datetime.strptime('1979-01-01', '%Y-%m-%d')

# Get the latest downloaded data
files00 = listdir(downloadDir)
files = fnmatch.filter(files00, '*multi*.nc')
files.sort()


def getDateF(f):
    endChar = f[-13:-3]
    endDate = datetime.strptime(endChar, '%Y-%m-%d')
    return endDate


endDates = map(getDateF, files)

try:
    lastDownloaded = max(endDates)
except:
    lastDownloaded = datetime.strptime('1978-12-31', '%Y-%m-%d')

from_day = lastDownloaded.date() + relativedelta(days=1)
# Construct the date of the last available data on ECMWF MARS server
now = datetime.now().date()
to_day = datetime(now.year, now.month, 1).date() - \
    relativedelta(months=2) - relativedelta(days=1)

if from_day < to_day:
    t_interval = "%s to %s" % (from_day, to_day)
    logging.info("Downloading data for the period %s..." % (t_interval))
    getMultivarMon(from_day=from_day, to_day=to_day,
                downloadDir=downloadDir, server=ECMWFDataServer())
else:
    logging.info("No need to download new data. Latest data available on the server is %s and data to download would be %s." % (
        to_day, from_day))

logging.info('JOB DONE')
