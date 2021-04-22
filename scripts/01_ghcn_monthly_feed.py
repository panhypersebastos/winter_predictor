import logging
import sys


def main():
    '''
    Main code to run in script mode
    '''
    sys.path.append('../')
    from pred.ghcn import GHCN

    G = GHCN(config_file='../data/config.json')
    G.wgetData()
    # HERE ------- !!!
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
