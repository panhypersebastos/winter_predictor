import logging
import os
from os import listdir
import pandas as pd
import datetime
from pymongo import MongoClient
import numpy as np


class GHCN():
    '''
    Class for downloading, ingesting and updating
    GHCN monthly data.
    '''

    remote_data = 'ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/v4/ghcnm.tavg.latest.qcu.tar.gz'

    def __init__(self,
                 downloadDir,
                 logfilename):
        '''
        downloadDir -- string Path where the GHCN data is saved
        '''
 
        # Stuff that get initialized
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
        Downloads the GHCN monthly data using wget
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
        Once the metadta file for stations has been downloaded,
        insert in the station collection.
        '''
        # Country metadata
        country_df = pd.read_fwf('ghcnm-countries.txt',
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


def main():
    '''
    Main code to run in script mode
    '''
    G = GHCN(downloadDir='/home/dmasson/data/GHCNM/',
             logfilename='/home/dmasson/temp/ghcnm.log')
    G.wgetData()
    logging.info('Job done.')


if __name__ == '__main__':
    main()
