from pymongo import MongoClient
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import cdsapi
import logging
from cputils.mongo import connect_mongo
import os
import os.path
import pymongo
import xarray as xr
from multiprocessing.pool import ThreadPool
from multiprocessing_logging import install_mp_handler
import pandas as pd
import itertools

def main():

    ERA = ERA5T(
        downloadDir='/home/dmasson/data/ERA5T_WP/',
        logfilename='/home/dmasson/data/ERA5T_WP/logfiles/era5.log',
        historical=True,
        download=True)

    if ERA.historical is True:
        years = np.arange(2000, datetime.today().year+1).tolist()
        # ERA.createDataColIndex()  # uncomment if the col does not exist yet

    else:
        today = datetime.today().date()
        years = np.arange(ERA.newday.year, today.year+1).tolist()
        logging.info('Latest date present in MongoDB: %s' % ERA.lastDate)
        logging.info('Updating MongoDB from  %s' % ERA.newday)

    logging.info('Years to process: %s' % years)
    ERA.processYears(years=years)
    logging.info(' --- JOB DONE !!! ---')


if __name__ == '__main__':
    main()
