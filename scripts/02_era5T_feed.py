# Copyright (C) David Masson <panhypersebastos@gmail.com>

import sys
import numpy as np
from datetime import datetime
import logging


def main(download: bool) -> None:
    # first_insertion: bool) -> None:
    '''
    Main code to run in script mode.

    Parameters
    ----------
    download : bool
        Should the data be downloaded ?
    # first_insertion : bool
    #     Is this the first insertion? If yes, a grid collection
    #     will be created.
    '''
    sys.path.append('../')
    from pred.era5T import ERA5T

    ERA = ERA5T(config_file='../data/config.json', ncpu=6, download=True)
    years = np.arange(ERA.newday.year, datetime.now().year+1).tolist()
    logging.info('Updating MongoDB from  %s' % ERA.newday)
    ERA.processYears(years=years)
    logging.info(' --- JOB DONE !!! ---')


if __name__ == '__main__':
    main()
