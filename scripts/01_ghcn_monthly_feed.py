# Copyright (C) David Masson <panhypersebastos@gmail.com>
import logging
import sys


def main(first_insertion: bool) -> None:
    '''
    Main code to run in script mode.

    Parameters
    ----------
    first_insertion : bool
        Is this the first insertion? If yes, a station collection
        needs to be created.
    '''
    sys.path.append('../')
    from pred.ghcn import GHCN

    G = GHCN(config_file='../data/config.json')
    G.wgetData()
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
    main(first_insertion=True)
