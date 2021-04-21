import logging
import os
from os import listdir
import pandas as pd
import datetime
from pymongo import MongoClient
import pymongo
import numpy as np

def main():
    '''
    Main code to run in script mode
    '''
    G = GHCN(downloadDir='/home/dmasson/data/GHCNM/',
             logfilename='/home/dmasson/temp/ghcnm.log')
    G.wgetData()
    G.insertDataCollection()

    historical = False
    if historical is True:
        # Generally done for the first insertion
        # Create the station collection (only needed once)
        G.upsertStationCollection()
        # Create indexes for station metadata
        G.createStationIndexing()
        # Create indexes for observations
        G.createDataIndexing()
    logging.info('Job done.')


if __name__ == '__main__':
    main()
