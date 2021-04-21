import logging
import os
from os import listdir
import pandas as pd
import datetime
from pymongo import MongoClient
import pymongo
import numpy as np


class GHCN():
    '''
    Class for downloading, ingesting and updating
    GHCN monthly data.
    '''

    remote_data = 'https://www1.ncdc.noaa.gov/pub/data/ghcn/v4/ghcnm.tavg.latest.qcu.tar.gz'

    def __init__(self,
                 downloadDir,
                 logfilename):
        '''
        downloadDir -- string Path where the GHCN data is saved
        '''

        self.downloadDir = downloadDir
        self.logfilename = logfilename
 
        # Logging setup
        # Remove all handlers associated with the root logger object
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if os.path.exists(self.logfilename):
            os.remove(self.logfilename)
        logging.basicConfig(
            filename=logfilename,
            format='%(asctime)s %(message)s',
            level=logging.INFO)
        logging.info('GHCN MONTHLY: job started')

    def wgetData(self):
        '''
        Downloads the GHCN monthly data using wget.
        This downloads both the station metadata and
        the monthly data itself.
        '''
        wget_command = 'wget -nH --cut-dirs=100 -np -P %s -m %s && tar -xzf %s/ghcnm.tavg.latest.qcu.tar.gz -C %s' % (
            self.downloadDir,
            self.remote_data,
            self.downloadDir,
            self.downloadDir)
        logging.info('Executing: %s' % wget_command)
        os.system(wget_command)
        logging.info('Executing: %s DONE' % wget_command)

    @staticmethod
    def _createMongoConn():
        # MongoDB connections
        mongo_host_local = 'mongodb://localhost:27017/'
        mg = MongoClient(mongo_host_local)
        db_name = 'GHCNM'
        col_sta = mg[db_name]['stations']
        col_dat = mg[db_name]['data']
        return({'con': mg,
                'col_sta': col_sta,
                'col_dat': col_dat})

    def createStationIndexing(self):
        '''
        Add indexes
        Warning: geospatial index require -180, +180 longitudes
        '''
        col_sta = self._createMongoConn()['col_sta']
        col_sta.create_index([("station_id", pymongo.ASCENDING)])
        col_sta.create_index([("loc", pymongo.GEOSPHERE)])

    def createDataIndexing(self):
        '''
        Add indexes for the station data collection
        '''
        col_dat = self._createMongoConn()['col_dat']
        col_dat.create_index([("station_id", pymongo.ASCENDING)])
        col_dat.create_index([("year", pymongo.DESCENDING)])

    def findNewestFile(self, pattern):
        '''
        pattern: str one among '.inv' for station metadata, or
                               '.dat' for time series
        '''
        # List the latest file
        fl = [os.path.join(dp, f) for dp, dn, fn in
              os.walk(os.path.expanduser(self.downloadDir))
              for f in fn if f.endswith(pattern)]

        df = pd.DataFrame({'path': fl})
        df = df.assign(filedate=list(map(
            lambda x: datetime.datetime.strptime(x[-16:-8], '%Y%m%d'),
            df['path'])))
        # Pick the newest file
        df = df.sort_values(by='filedate',
                            ascending=False).reset_index(drop=True)
        path = df.loc[0, 'path']
        return(path)
 
    def upsertStationCollection(self):
        '''
        Once the metadta file for stations has been downloaded
        (generally in the same wget step as for the observations),
        insert station metadata in the station collection.
        '''
        # Country metadata
        country_df = pd.read_fwf('%sghcnm-countries.txt' % self.downloadDir,
                                 colspecs=[[0,2], [3, 500]],
                                 header=None,
                                 names=['country_id', 'country'])
        # Get station metadata
        path = self.findNewestFile(pattern='.inv')
        sta_df = pd.read_fwf(path,
                             colspecs=[[0,11], [0,2],[2,8],[13, 20],
                                       [24, 30], [31,37], [38,69],
                                       [90, 106],[106,107]],
                             header=None,
                             names=['station_id','country_id',
                                    'wmo_id', 'lat', 'lon', 'elev',
                                    'name', 'landcover', 'popclass'])
        sta_df = pd.merge(sta_df, country_df, on='country_id')

        # Upsert into MongoDB
        col_sta = self._createMongoConn()['col_sta']

        def upsertStation(i):
            newdoc = dict({
                'station_id': str(sta_df.station_id[i]),
                'loc': {'type': 'Point',
                        'coordinates': [float(sta_df.lon[i]),
                                        float(sta_df.lat[i])]},
                       'country': sta_df.country[i],
                       'country_id': str(sta_df.country_id[i]),
                       'wmo_id': str(sta_df.wmo_id[i]),
                       'elev': sta_df.elev[i],
                       'name': sta_df.name[i],
                       'landcover': sta_df.landcover[i],
                'popclass': sta_df.popclass[i]})

            col_sta.update_one(
                filter={"station_id": newdoc['station_id']},
                update=dict({'$set': newdoc}), upsert=True)
        void = list(map(upsertStation, np.arange(sta_df.shape[0])))

    def insertDataCollection(self):
        '''
        Insert the csv table "as is".
        We anyway need to group by month later in the analysis.
        This code delete any existing data and re-insert it.
        '''
        # Read the most recent file
        fnew = self.findNewestFile(pattern='.dat')
        logging.info('Inserting %s', fnew)
        dat_df = pd.read_fwf(fnew,
                             na_values='-9999',
                             colspecs=[[0, 11], [11, 15], [15, 19],
                                       [19, 24], [27, 32], [35, 40],
                                       [43, 48], [51, 56], [59, 64],
                                       [67, 72], [75, 80], [83, 88],
                                       [91, 96], [99, 104], [107, 112]],
                             header=None,
                             #nrows=20,
                             names=['station_id', 'year', 'variable',
                                    '1', '2', '3', '4', '5', '6',
                                    '7', '8', '9', '10', '11', '12'])
        # Convertion to °C
        dat_df['1'] = dat_df['1']/100
        dat_df['2'] = dat_df['2']/100
        dat_df['3'] = dat_df['3']/100
        dat_df['4'] = dat_df['4']/100
        dat_df['5'] = dat_df['5']/100
        dat_df['6'] = dat_df['6']/100
        dat_df['7'] = dat_df['7']/100
        dat_df['8'] = dat_df['8']/100
        dat_df['9'] = dat_df['9']/100
        dat_df['10'] = dat_df['10']/100
        dat_df['11'] = dat_df['11']/100
        dat_df['12'] = dat_df['12']/100
        # Insert the table above "as is".
        # We anyway need to group by month later in the analysis.
        col_dat = self._createMongoConn()['col_dat']
        # Delete and re-insert
        col_dat.delete_many(filter={})
        col_dat.insert_many(dat_df.to_dict('records'))


def main():
    '''
    Main code to run in script mode
    '''

if __name__ == '__main__':
    main()