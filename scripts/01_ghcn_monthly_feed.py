# Copyright (C) David Masson <panhypersebastos@gmail.com>
import logging
import sys


def main(download: bool, first_insertion: bool) -> None:
    '''
    Main code to run in script mode.

    Parameters
    ----------
    download : bool
        Should the data be downloaded ?
    first_insertion : bool
        Is this the first insertion? If yes, a station collection
        will be created.
    '''
    sys.path.append('../')
    from pred.ghcn import GHCN

    G = GHCN(config_file='../data/config.json')
    if download is True:
        G.wgetData(target='data')
        G.wgetData(target='countries')
    G.insertDataCollection()
    if first_insertion is True:
        # Generally done for the first insertion
        # Create the station collection (only needed once)
        G.upsertStationCollection()
        # Create indexes for station metadata
        G.createStationIndexing()
        # Create indexes for observations
        G.createDataIndexing()
    logging.info('Job done.')


if __name__ == '__main__':
    main(download=True, first_insertion=True)
