from pymongo import MongoClient
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import cdsapi
import logging
import os
import os.path
import pymongo
import xarray as xr
from multiprocessing.pool import ThreadPool
from multiprocessing_logging import install_mp_handler
import pandas as pd
import itertools
from json import loads
from typing import List
import tempfile
import pathlib


class ERA5T():
    '''
    Class for downloading, ingesting and updating
    ERA5T monthly data.
    '''

    start_era5t = datetime(year=1979, month=1, day=1)

    def __init__(self,
                 config_file: str,
                 nthreads: int = 6,
                 download: bool = False,
                 ) -> None:
        # downloadDir,
        # logfilename,
        # experimental_setting,
        # historical,
        # download) -> None:
        '''
        Initializes an instance of the "ERA5T" class.

        Parameters
        ----------
        config_file : str
            Path to the JSON file holding the MongoDB credentials and
            the paths where to store the data and logfile.
        nthreads : int
            Number of parallel processes. Per default 4 threads.
        download : bool
            Shall files be downloaded or just work with files already present.
            Default is False.

        Returns
        -------
        None
            The statement is executed without return value.
        '''
        with open("../data/config.json", "r", encoding="utf-8") as config:
            cfg = loads(config.read())
        self.downloadDir = cfg['download_dir'] + 'ERA5T/'
        self.logfilename = cfg['download_dir'] + 'era5t.log'
        self.cfg = cfg

        if os.path.exists(self.downloadDir) is False:
            try:
                from pathlib import Path
                Path(self.downloadDir).mkdir(parents=True, exist_ok=True)
            except OSError:
                print("Creation of the directory %s failed" %
                      self.downloadDir)
            else:
                print("Created the directory %s " % self.downloadDir)

        self.lastDate = self.getLastDate()
        #  Shall all historical files be downloaded? If True:
        #  * either the database is initialized for the first time
        #  * or you want to update already inserted data.
        # self.historical = (self.lastDate == self.start_era5t)
        # self.lsm = None  # Land mask
        self.download = download

        # In order to avoid loading complete year, add one day
        # The past 3 months are always up-2-date
        self.newday = self.lastDate + \
            relativedelta(days=1) - \
            relativedelta(months=3)
        if self.newday < self.start_era5t:
            self.newday = self.start_era5t

        # Logging setup
        # Remove all handlers associated with the root logger object
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if os.path.exists(self.logfilename):
            os.remove(self.logfilename)
        logging.basicConfig(
            filename=self.logfilename,
            format='%(asctime)s %(message)s',
            level=logging.INFO)
        logging.info('ERA5T Job started')

        # # Get all grid_ids
        # self.getAllGridIds()

    @ staticmethod
    def _createMongoConn(cfg: dict) -> dict:
        '''
        Establish a connection to MongoDB.

        Parameters
        ----------
        cfg : dict
            Dictionary containing credentials and ERA5T database name.

        Returns
        -------
        Dict
            A dictionary containing access MongoClient and access to
            ERA5T collections.
        '''
        # MongoDB connections
        db_host = cfg['db_host']
        db_user = cfg['db_user']
        db_password = cfg['db_password']
        db_port = cfg['db_port']
        con_string = f"mongodb://{db_user}:" + \
                     f"{db_password}@{db_host}:{db_port}/?authSource=admin"
        mg = MongoClient(con_string)
        db_name = cfg['db_ERA5T_name']
        col_grid = mg[db_name]['grid']
        col_dat = mg[db_name]['data']
        return({'con': mg,
                'col_sta': col_grid,
                'col_dat': col_dat})

    def getLandMask(self) -> None:
        '''
        This function downloads the ERA5 land mask
        and is used only once. It is relavant if the grid is limited to land
        area and is used in the function "createLandGridCollection".
        (Function not relevant for the winter predictor project).
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
            target=f'{self.downloadDir}land_sea_mask.nc')

        logging.info('Downloading land mask DONE.')

    def getAllGridIds(self):
        '''
        This function returns all existing grid ids
        '''
        con = connect_mongo(prod=True, rw=False)
        col_grid = con['ECMWF']['ERA5_grid']
        self.all_ids = col_grid.distinct(key='id_grid')

    def getFiles(self, year: int, month: int) -> None:
        '''
        Loads data via the API, merge files into a single NetCDF files
        and store the result on disk.

        Parameters
        ----------
        year : int
        month : int

        # Returns
        # -------
        # List[str]
        #     A list containing two paths:
        #     * One pointing to the data at pressure levels (z70).
        #     * The other pointing to the data at single pressure level (sst, t2m, etc.).
        '''
        # Get inspiration from :
        # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=form
        # and also
        # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means?tab=form

        # Geopotentials heights data
        filename = f'era5_{year}-{month}'
        logging.info("PROCESSING %s..." % (filename))
        tempdir = tempfile.TemporaryDirectory().name
        pathlib.Path(tempdir).mkdir(parents=True, exist_ok=True)
        temp_path = f'{tempdir}/{filename}'
        path = f'{self.downloadDir}{filename}.nc'

        c = cdsapi.Client()
        # Data at pressure levels:
        f01 = temp_path + '_part01.nc'
        c.retrieve(
            'reanalysis-era5-pressure-levels-monthly-means',
            {
                'format': 'netcdf',
                # Latitude/longitude grid, default: 0.25 x 0.25
                'grid': [1.0, 1.0],
                'variable': [
                    'geopotential'],
                'pressure_level': '70',
                'year': int(year),
                'month': int(month),
                'time': '00:00',
                'product_type': 'monthly_averaged_reanalysis'
            },
            f01)
        # Data at single pressure level:
        f02 = temp_path + '_part02.nc'
        c.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {
                'format': 'netcdf',
                # Latitude/longitude grid, default: 0.25 x 0.25
                'grid': [1.0, 1.0],
                'product_type': 'monthly_averaged_reanalysis',
                'variable': [
                    'sea_ice_cover',
                    'surface_pressure',
                    'sea_surface_temperature',
                    '2m_temperature',
                    'mean_sea_level_pressure',
                    'total_precipitation',
                ],
                'year': int(year),
                'month': int(month),
                'time': '00:00'
            },
            f02)
        try:
            ds01 = xr.open_dataset(f01)
            ds02 = xr.open_dataset(f02)
        except Exception as e:
            print(e)
            logging.info('ERROR: could not process %s' % filename)
        else:
            ds = ds02.merge(ds01)
            ds.to_netcdf(path)

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

    def createLandGridCollection(self) -> None:
        '''
        Creation of a grid collection limited to land surface,
        excluding sea surface.
        (Function not relevant for the winter predictor project).
        '''
        logging.info('INGESTION of the grid collection started...')
        # Open ERA5 land mask (downloaded with the "getLandMask" function.)
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

    def getLastDate(self) -> datetime:
        '''
        Get the last date of ERA5T data stored in MongoDB.

        Parameters
        ----------
        None

        Returns
        -------
        datetime
        '''
        con = self._createMongoConn(cfg=self.cfg)
        col_dat = con['col_dat']
        n = col_dat.count()
        if n != 0:
            doc = col_dat.find_one(
                sort=[('year', pymongo.DESCENDING)])
            ndoy = len(doc['t2m'])
            last = datetime.strptime('%s %s' % (doc['year'], ndoy), '%Y %j')

        else:
            logging.info('No data present in MongoDB yet.')
            last = self.start_era5t
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

    def findMissingDates(self, ds: xr.core.dataset.Dataset) \
            -> pd.core.frame.DataFrame:
        '''
        Finds the missing dates in an xarray dataset object and
        returns a pd.DataFrame containing the missing dates in the
        'time' column.
        This functions only makes sense for dataset with daily time series.

        Parameters
        ----------
        ds : xr.core.dataset.Dataset

        Returns
        -------
        pd.core.frame.DataFrame
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

        # p = ThreadPool(processes=self.nthreads, initializer=worker_initializer)
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

    def exploreChunks(self, ilon_chunk: int, ilat_chunk: int, delta: int,
                      # HERE !!!!
                      retrn: str, col_grid):
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

    def listNetCDFfiles(self, year: int) -> List[str]:
        '''
        List all nc files belonging to a given year.

        Parameters
        ----------
        year : int

        Returns
        -------
        List[str]
            A list containing the path to the targeted nc files.
        '''
        nc_local = ['%s/%s' % (self.downloadDir, f) for f in
                    os.listdir(self.downloadDir) if
                    f.startswith(f'era5_{year}') and f.endswith('.nc')]
        return(nc_local)

    def processYears(self, years: List[int]) -> None:
        '''
        Parameters
        ----------
        years : List[int]
            list of years to (re-)process
        '''
        for year in years:
            self.year = int(year)
            logging.info(' --- PROCESSING YEAR %s ---' % year)

            if self.download is True:
                logging.info('Proceeding with downloads...')
                today = datetime.today()
                if (year == today.year):
                    months = np.arange(1,
                                       today.month + 1).tolist()
                else:
                    months = np.arange(1, 12 + 1).tolist()

                # Are these months present as nc files ?
                # List this year's nc files
                try:
                    ncfiles = self.listNetCDFfiles(year)
                except FileNotFoundError:
                    print(f'No ERA5T files downloaded for {year} yet.')
                    months_to_download = months
                else:
                    # Probably a bug HERE !!! To be tested.
                    fmonths_present = sorted(list(
                        map(lambda x: int(x[x.find("-")+1: x.find(".nc")]),
                            ncfiles)))
                    fmonths_needed = months
                    # Months needed but not present :
                    missing_months = list(set(fmonths_needed) -
                                          (set(fmonths_present)))
                    months_to_download = list(
                        set(missing_months + months))  # distinct months

                logging.info(
                    f'Downloading files for YEAR {year}....\n Months: {months_to_download}')
                # Parralel download of monthly data:
                install_mp_handler()
                p = ThreadPool(processes=self.nthreads)
                p.map(lambda m: self.getFile(year=year, month=m),
                      months_to_download)
                p.close()
                p.join()
                logging.info(f'Downloading files for YEAR {year} Done.')
            else:
                logging.info('Proceeding without downloads')

            # List all the current year's nc files after download
            nc_local = self.listNetCDFfiles(year=year)

            # Open them all in one ds object
            # arrays will be loaded in chronological order
            ds = xr.open_mfdataset(nc_local, combine='by_coords')
            # only makes sense for daily time-series:
            # self.df_missing_dates = self.findMissingDates(ds)

            # Create the tile (chunks) elements
            # This operation starts to be useful at high grid resolution
            # i.e., from 0.25 x 0.25. For coarser grid (i.e., 0.1 x 0.1)
            # this is not really vital.
            delta = 10  # grid chunk in degrees
            # ERA's lon have range [0, 360] and not [-180, 180]
            ilons = np.arange(0, 360, delta)
            ilats = np.arange(-60, 90, delta)
            elements = itertools.product(*[ilons, ilats])

            # Explore the grid chunks and select those containing grid cells
            def worker_initializer00():
                global col_grid
                cons = self._createMongoConn(cfg=self.cfg)
                col_grid = cons['col_grid']

            p = ThreadPool(processes=self.nthreads,
                           initializer=worker_initializer00)
            # HERE !!
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
