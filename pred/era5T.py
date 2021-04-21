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


class ERA5T():
    '''
    Get ERA5 data using ECMWF's API (CDSAPI).
    This modules downloads hourly data,
    aggregate it into daily data and
    ingest it into MongoDB.
    Main steps:
    - get remote hourly data for a given month using CDSAPI
    - aggregate houlry to daily
    - concatenate netCDF files into one yearly file
    - insert it into the data collection with the following structure:
      {'_id': ObjectId('5d64a634bf7df96b6d6de5fa'),
       'id_grid': 155333,
       'year': 2001,
       't2m: [-1.2, -3.5, 0.5, 5, ...],
       'tp': [0, 0, 4, 5, 6.5, 0, ...]}
    '''
    ncpu = 6

    def __init__(self,
                 downloadDir,
                 logfilename,
                 experimental_setting,
                 historical,
                 download):

        self.downloadDir = downloadDir
        self.logfilename = logfilename
        self.historical = historical
        self.download = download
        self.lsm = None  # Land mask

        self.lastDate = self.getLastDate()
        # In order to avoid loading complete year, add one day
        # The past 3 months are always up-2-date
        self.newday = self.lastDate + \
            relativedelta(days=1) - \
            relativedelta(months=3)

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
        logging.info('ERA5T Job started')

        # Get all grid_ids
        self.getAllGridIds()

    @staticmethod
    def _createMongoConn():
        # MongoDB connections
        # con = connect_mongo(prod=operator.not_(experimental_setting),
        #                    rw=True)

        cfg_MONGO_CLIENT = MongoClient('mongodb://localhost:27017/')

        con = cfg_MONGO_CLIENT
        col_grid = con['ECMWF']['ERA5_grid']
        col_dat = con['ECMWF']['ERA5_data']
        return({'con': con,
                'col_grid': col_grid,
                'col_dat': col_dat})

    def getLandMask(self):
        '''
        This function downloads the ERA5 land mask
        and is used only once.
        '''
        # Look there:
        # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form
        logging.info('Downloading land mask...')
        c = cdsapi.Client()
        c.retrieve(
            name='reanalysis-era5-single-levels',
            request={
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'land_sea_mask',
                    'sea_surface_temperature'
                ],
                'year': '1979',
                'month': '01',
                'day': '01',
                'time': '00:00'
            },
            target='%s/land_sea_mask.nc' % self.downloadDir)

        logging.info('Downloading land mask DONE.')

    def getAllGridIds(self):
        '''
        This function returns all existing grid ids
        '''
        con = connect_mongo(prod=True, rw=False)
        col_grid = con['ECMWF']['ERA5_grid']
        self.all_ids = col_grid.distinct(key='id_grid')

    def getFiles(self, year, month):  # CONTINUE HERE !!!
        # Get inspiration from https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=form

        # get MULTIPLE VARIABLES :
        filename01 = 'era5_file01_%s_%s.nc' % (year, month)
        filename02 = 'era5_file02_%s_%s.nc' % (year, month)
        filename03 = 'era5_file03_%s_%s.nc' % (year, month)

        logging.info("PROCESSING %s..." % (filename01))
        # Here comes variables like temperature ...
        # server.retrieve({
        #     "class": "ei",
        #     "dataset": "interim",
        #     "date": datestring,
        #     "expver": "1",
        #     "grid": "%s/%s" % (resolution, resolution),
        #     "levtype": "sfc",
        #     "param": "31.128/34.128/35.128/134.128/139.128/151.128/165.128/166.128/167.128/168.128/174.128/186.128/187.128/188.128/207.128/235.128",
        #     "stream": "moda",
        #     "type": "an",
        #     "format": "netcdf",
        #     "target": "%s/%s" % (downloadDir, filename01),
        # })

        # Here comes ?? variables ??
        # logging.info("PROCESSING %s..." % (filename02))
        # server.retrieve({
        #     "class": "ei",
        #     "dataset": "interim",
        #     "date": datestring,
        #     "expver": "1",
        #     "grid": "%s/%s" % (resolution, resolution),
        #     "levtype": "sfc",
        #     "param": "159.128/231.128/232.128",
        #     "stream": "moda",
        #     "format": "netcdf",
        #     "type": "fc",
        #     "target":  "%s/%s" % (downloadDir, filename02),
        # })

        # Geopotentials heights data
        logging.info("PROCESSING %s..." % (filename03))
        path03 = '%s/downloads/%s' % (self.downloadDir, filename03)
        server.retrieve({'reanalysis-era5-pressure-levels-monthly-means',
                         'format': 'netcdf',
                         'variable': 'geopotential',
                         'pressure_level': '700',
                         'year': int(year),
                         'month': int(month),
                         'time': '00:00',
                         'product_type': 'monthly_averaged_reanalysis',
                         },
                        path)

        #     "class": "ei",
        #     "dataset": "interim",
        #     "date": datestring,
        #     "expver": "1",
        #     "grid": "%s/%s" % (resolution, resolution),
        #     "levelist": "70",
        #     "levtype": "pl",
        #     "param": "129.128",
        #     "stream": "moda",
        #     "type": "an",
        #     "format": "netcdf",
        #     "target":  "%s/%s" % (downloadDir, filename03),
        # })

    def getFile(self, year, month):  # DEPRECATED
        '''
        Download ERA5 netCDF file. Note that a monthly file is ~ 3GB large.
        '''
        targetfile = 'era5_%s-%s.nc' % (year, month)
        path = '%s/%s' % (self.downloadDir, targetfile)
        if os.path.isfile(path) is True:
            logging.info('%s Already exists, deleting...' % targetfile)
            os.remove(path)

        logging.info('PROCESSING %s' % targetfile)

        vars = [
            'sea_ice_cover',  # ci (0-1)
            '2m_temperature',  # 2t [K]
            'total_precipitation',  # tp [m]
            'sea_surface_temperature',  # sst [K]
            'ice_temperature_layer_1',  # istl1 [K]
            'surface_pressure',  # sp [Pa]
            'soil_temperature_level_1',  # stl1 [K]
            'mean_sea_level_pressure',  # msl [Pa]
            '2m_dewpoint_temperature',  # 2d [K]
            '10m_wind_gust_since_previous_post_processing',  # 10gf [m/s]
            # geopotential height -- specify the level ! z [m/s^2]
            'geopotential'
        ]

        c = cdsapi.Client()
        try:
            c.retrieve(name='reanalysis-era5-single-levels',
                       request={
                           'product_type': 'reanalysis',
                           'format': 'netcdf',
                           'variable': vars,
                           "levelist": "700",
                           'year': int(year),
                           'month': int(month),
                           'day': np.arange(1, 31 + 1).tolist(),
                           'time': [
                               '00:00', '01:00', '02:00',
                               '03:00', '04:00', '05:00',
                               '06:00', '07:00', '08:00',
                               '09:00', '10:00', '11:00',
                               '12:00', '13:00', '14:00',
                               '15:00', '16:00', '17:00',
                               '18:00', '19:00', '20:00',
                               '21:00', '22:00', '23:00'
                           ]
                       },
                       target=path)
        except Exception as e:
            print(e)
        logging.info(
            'DOWNLOAD of %s DONE.\n Transforming file from hourly to daily...' % targetfile)

        # Resample hourly to daily and save
        try:
            ds = xr.open_dataset(path)
        except Exception as e:
            print(e)
            logging.info('ERROR: could not process %s' % targetfile)
        else:
            ds_agg = ds.resample(time='D').mean()
            ds_agg.to_netcdf(path)

    def createDataColIndex(self):
        '''
        Create indexes for the data collection
        '''
        index1 = pymongo.IndexModel([("year", pymongo.DESCENDING)],
                                    name="year_-1")
        index2 = pymongo.IndexModel(
            [("id_grid", pymongo.ASCENDING),
             ("year", pymongo.DESCENDING)],
            name="id_grid_1_year_-1")
        con = self._createMongoConn(self.experimental_setting)
        col_dat = con['col_dat']
        col_dat.create_indexes([index1, index2])
        logging.info('Indexes added for the data collection')

    @staticmethod
    def shiftlon(x):
        if x > 180:
            x = x - 360
        return(x)

    @staticmethod
    def shiftback_lon(x):
        if x < 0:
            x = x + 360
        return(x)

    def createGridCollection(self):
        '''
        Creation of the grid collection
        '''
        logging.info('INGESTION of the grid collection started...')
        # Open ERA5 land mask
        f = '%s/land_sea_mask.nc' % self.downloadDir
        ds = xr.open_dataset(f)
        # Limit the grid collection to land only and exclude Antarctica
        # LSM is the proportion of land/sea in a grid box
        field = ds.isel(time=0).where((ds.lsm.isel(time=0) > 0) &
                                      (ds.latitude >= -60))
        df = field.to_dataframe().reset_index()

        def createCoord(lon, lat):
            newlon = self.shiftlon(lon)
            res = {'type': 'Point',
                   'coordinates': [newlon, lat]}
            return(res)

        df = df.assign(id_grid=np.arange(df.shape[0]),
                       loc=list(map(lambda lon, lat: createCoord(lon, lat),
                                    df['longitude'],
                                    df['latitude'])))
        df = df.drop(columns=['time', 'sst', 'longitude', 'latitude'])
        df = df.dropna()

        con = self._createMongoConn(self.experimental_setting)
        col_grid = con['col_grid']
        col_grid.insert_many(df.to_dict('records'))
        logging.info('INGESTION of the grid collection DONE.')

    def createGridColIndex(self):
        '''
        Create Geo-Spatial indexes
        Warning: geospatial index require -180, +180 longitudes.
        '''
        con = self._createMongoConn(self.experimental_setting)
        col_grid = con['col_grid']
        col_grid.create_index([("loc", pymongo.GEOSPHERE),
                               ("id_grid", pymongo.ASCENDING)])
        logging.info('Indexes added for the grid collection')

    def getLastDate(self):
        '''
        Get the last stored date from MongoDB
        '''
        con = self._createMongoConn(self.experimental_setting)
        col_dat = con['col_dat']
        n = col_dat.count()
        if n != 0:
            doc = col_dat.find_one(
                sort=[('year', pymongo.DESCENDING)])
            ndoy = len(doc['t2m'])
            last = datetime.strptime('%s %s' % (doc['year'], ndoy), '%Y %j')

        else:
            logging.info('No data present in MongoDB yet.')
            last = datetime(year=1979, month=1, day=1)
        return(last)

    def createRow(self, doc, df_missing_dates, DS):
        '''
        Create a dataframe ready to be inserted into MongoDB
        in the data collection
        ---
        doc -- dictionary Comes from the query on the grid collection
        df_missing_dates -- df Comes from the call to 'findMissingDates'
        DS -- xarrax dataset
        '''
        this_year = int(self.year)
        id_grid = int(doc['id_grid'])
        lon, lat = doc['loc']['coordinates']
        lon = self.shiftback_lon(x=lon)

        df0 = DS.sel({'longitude': lon, 'latitude': lat},
                     method='nearest').to_dataframe().reset_index()
        df0['time'] = list(map(lambda x: pd.to_datetime(x).date(),
                               df0['time']))

        if (df_missing_dates.shape[0] > 0):
            # Fill with NA days
            df0 = pd.merge(left=df0,
                           right=df_missing_dates,
                           on='time', how='outer').sort_values(
                               by='time', ascending=True)
        t2m_trans = list(map(
            lambda x: round(x - 273.15, 2),  # from °K into °C
            df0['t2m']))
        tp_trans = list(map(
            lambda x: round(x * 1000 * 24, 2),  # from m/h into mm/day,
            df0['tp']))
        update_doc = {'year': this_year,
                      'id_grid': id_grid,
                      'tp': tp_trans,
                      't2m': t2m_trans}
        return(update_doc)

    def findMissingDates(self, ds):
        '''
        Finds the missing dates in an xarray object and return a pd.DataFrame
        ds -- xarray
        '''
        this_year = self.year
        dates_df = pd.DataFrame(
            {'recorded': list(map(lambda x: pd.to_datetime(x).date(),
                                  ds['time'].values.tolist()))}
        ).sort_values(by='recorded')
        mindate = datetime(year=this_year, month=1, day=1)
        maxdate = dates_df[['recorded']].max()[0]
        # Construct dataframe with all days inside
        df_fill = pd.DataFrame({
            'date': pd.date_range(mindate, maxdate, name='time')})
        df_fill['date'] = list(map(lambda x: pd.to_datetime(x).date(),
                                   df_fill['date']))
        # Avoid any gaps
        data_df = pd.merge(left=df_fill,
                           right=dates_df,
                           left_on='date',
                           right_on='recorded',
                           how='outer').sort_values(by='date', ascending=True)
        missing_dates = data_df.query('recorded.isnull()')['date']
        df_missing_dates = pd.DataFrame({'time': missing_dates})
        df_missing_dates['time'] = list(map(lambda x: pd.to_datetime(x).date(),
                                            df_missing_dates['time']))
        return(df_missing_dates)

    def insertChunk(self, ilon_chunk, ilat_chunk, delta, ds, method):
        '''
        A chunk is a small xarray subset that can be loaded in memory.
        This function inserts or upserts all the grid cells data
        that are within the chunk's area.
        ---
        ilon_chunk -- int Longitude of the upper-left bounding box corner
        ilon_chunk -- int Latitude of the upper-left bounding box corner
        delta -- int width and height of the bounding box (in degrees)
        ds -- xarray The complete NetCDF raster for this year
        method -- string Insertion method, one of ['insert', 'upsert']
        '''
        ds_chunk = ds[['t2m', 'tp']].sel(
            longitude=slice(ilon_chunk, ilon_chunk + delta),
            latitude=slice(ilat_chunk + delta,
                           ilat_chunk))  # because lat are decreasing
        # load raster into memory
        # ds_chunk = ds_chunk.load()

        # Get the gridcells
        # Create MongoDB query
        con = self._createMongoConn(
            experimental_setting=self.experimental_setting)
        col_grid = con['col_grid']
        grid_docs = self.exploreChunks(
            ilon_chunk, ilat_chunk, delta, 'docs', col_grid)

        # Parralel does not seem to work properly...
        # # A connection will be created for each parallized thread:
        # def worker_initializer():
        #     global DS
        #     DS = ds_chunk

        # p = ThreadPool(processes=self.ncpu, initializer=worker_initializer)
        # res = p.map(
        #     lambda x: self.createRow(x, self.df_missing_dates, DS),
        #     grid_docs)
        # p.close()
        # p.join()
        res = list(map(
            lambda x: self.createRow(x, self.df_missing_dates, ds_chunk),
            grid_docs))

        col_dat = con['col_dat']
        if (method == 'insert'):
            df = pd.DataFrame(res)
            ids = df['id_grid'].values.tolist()
            # First remove data
            col_dat.delete_many(filter={'year': int(self.year),
                                        'id_grid': {'$in': ids}})
            # Then insert data
            col_dat.insert_many(df.to_dict('records'))
        else:
            logging.info('Upsert not implemented yet')

    def exploreChunks(self, ilon_chunk, ilat_chunk, delta, retrn, col_grid):
        '''
        Explore an xarray chunk and returns either the number
        of grid cells or the grid ids.
        ----
        ilon_chunk -- int Longitude of the upper-left bounding box corner
        ilon_chunk -- int Latitude of the upper-left bounding box corner
        delta -- int width and height of the bounding box (in degrees)
        retrn -- string What to return, one of ['ndocs', 'docs']
        col_grid -- mongo connection to the grid collection
        '''
        ilon_orig = int(ilon_chunk)
        ilon_chunk = int(self.shiftlon(x=ilon_chunk))
        ilon_plus = int(self.shiftlon(x=ilon_chunk + delta))
        ilat_chunk = int(ilat_chunk)
        geoqry = {'loc':
                  {'$geoWithin':
                   {'$geometry':
                    {'type': 'Polygon',
                     'coordinates': [[
                         [ilon_chunk, ilat_chunk],
                         [ilon_plus, ilat_chunk],
                         [ilon_plus, ilat_chunk + delta],
                         [ilon_chunk, ilat_chunk + delta],
                         [ilon_chunk, ilat_chunk]
                     ]]}}}}
        if retrn == 'ndocs':
            # How many grid cells in this chunk ?
            res = {'ilon_chunk': ilon_orig,
                   'ilat_chunk': ilat_chunk,
                   'n': col_grid.count(geoqry)}
        elif retrn == 'docs':
            res = col_grid.find(geoqry, {'_id': 0})
        return(res)

    def listNetCDFfiles(self, year):
        '''
        List all the current year's nc files
        ---
        year -- int
        '''
        nc_local = ['%s/%s' % (self.downloadDir, f) for f in
                    os.listdir(self.downloadDir) if
                    f.startswith('era5_%s' % year) and f.endswith('.nc')]
        return(nc_local)

    def processYears(self, years):
        '''
        years -- list of years to (re-)process
        '''
        for year in years:
            self.year = int(year)
            logging.info(' --- PROCESSING YEAR %s ---' % year)

            if self.download is True:
                today = datetime.today()
                if year == today.year:
                    months = list(
                        np.arange(self.newday.month, today.month + 1))
                else:
                    months = list(np.arange(self.newday.month, 12 + 1))

                # Are these months present as nc files ?
                # List this year's nc files
                ncfiles = self.listNetCDFfiles(year)
                fmonths_present = sorted(list(
                    map(lambda x: int(x[x.find("-")+1:x.find(".nc")]),
                        ncfiles)))
                fmonths_needed = list(np.arange(1, months[0]))
                # Months needed but not present :
                missing_months = list(set(fmonths_needed) -
                                      (set(fmonths_present)))
                months_to_download = list(
                    set(missing_months + months))  # distinct months

                logging.info(
                    'Downloading files for YEAR %s....\n Months: %s' % (
                        year,
                        months_to_download))
                # Parralel download of monthly data:
                install_mp_handler()
                p = ThreadPool(processes=self.ncpu)
                p.map(lambda m: self.getFile(year=year, month=m),
                      months_to_download)
                p.close()
                p.join()
                logging.info('Downloading files for YEAR %s Done.' % year)
            else:
                logging.info('Proceeding without downloads')

            # List all the current year's nc files after download
            nc_local = self.listNetCDFfiles(year=year)

            # Open them all in one ds object
            # arrays will be loaded in chronological order
            ds = xr.open_mfdataset(nc_local, combine='by_coords')
            self.df_missing_dates = self.findMissingDates(ds)

            # Create the tile (chunks) elements
            delta = 10  # in degrees
            # ERA's lon have range [0, 360] and not [-180, 180]
            ilons = np.arange(0, 360, delta)
            ilats = np.arange(-60, 90, delta)
            elements = itertools.product(*[ilons, ilats])

            # Explore the chunks and select those containing grid cells
            def worker_initializer00():
                global col_grid
                cons = self._createMongoConn(
                    experimental_setting=self.experimental_setting)
                col_grid = cons['col_grid']

            p = ThreadPool(processes=self.ncpu,
                           initializer=worker_initializer00)
            res = p.map(
                lambda e: self.exploreChunks(
                    e[0], e[1], delta, 'ndocs', col_grid),
                elements)
            p.close()
            p.join()
            df_e = pd.DataFrame(res)
            df_e = df_e.query(
                'n > 0').sort_values(by='n').reset_index(drop=True)

            # Do the insertion
            N = df_e.shape[0]
            for i in np.arange(N):
                logging.info('Year %s: processing chunk %s/%s' % (year, i, N))
                ilon = df_e.loc[i, 'ilon_chunk']
                ilat = df_e.loc[i, 'ilat_chunk']
                n = df_e.loc[i, 'n']
                self.insertChunk(ilon, ilat, delta, ds, 'insert')
                logging.info('%s documents inserted' % n)
            logging.info(' --- PROCESSING YEAR %s DONE !---' % year)


def main():
    '''
    Main code to run in script mode
    '''

if __name__ == '__main__':
    main()
