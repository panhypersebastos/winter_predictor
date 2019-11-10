import logging
import os
from pymongo import MongoClient


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
    def _createMongoConn(experimental_setting):
        # MongoDB connections
        # con = connect_mongo(prod=operator.not_(experimental_setting),
        #                    rw=True)
        if experimental_setting is True:
            cfg_MONGO_CLIENT = MongoClient(
                ["mongodb://bla:blabla@your_server:your_port"])
 
        con = cfg_MONGO_CLIENT
        col_grid = con['your_db']['your_grid']
        col_dat = con['your_db']['your_data']
        return({'con': con,
                'col_grid': col_grid,
                'col_dat': col_dat})
 
    def your_function(self, x):
        '''
        Function description
        ---
        x -- type Description
        '''
        y = x+1
        return(y)


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
