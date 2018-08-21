# This file contains all relevant code
# in object-oriented programming style

import numpy as np
import logging
import pymongo
import pandas as pd
from sklearn.decomposition import PCA
import sklearn.linear_model as skl_lm
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import os
import sys


class Predictor:
    ''' This class is about gathering all the predictors as defined by
    Wang et al. (2017). The central output is a DataFrame looking like:
    |WinterYear | PC1_predA_month1 | ... | PC3_predZ_month2|
    Generally, 'month1' and 'month2' are November and December.
    The Predictor object will be the same for whatever station
    to be predicted.'''

    def __init__(self):
        cons = Predictor._initializeCon()
        self.con_anom = cons['con_anom'],  # ERAINT anomalies collection
        self.con_grid = cons['con_grid'],  # ERAINT grid collection
        self.anom_df = pd.DataFrame()  # anomalies for a set of grid cells
        self.X_df = pd.DataFrame()  # final DataFrame of predictors

    @staticmethod
    def _initializeCon(ERA_vers='lores'):
        ''' Create the connections to the MongoDB collections '''
        if (ERA_vers == 'hires'):
            col_dat = 'ERAINT_monthly'
            col_anom = 'ERAINT_monthly_anom'
            col_grid = 'ERAINT_grid'
            resolution = 0.25
        elif (ERA_vers == 'lores'):
            col_dat = 'ERAINT_lores_monthly'
            col_anom = 'ERAINT_lores_monthly_anom'
            col_grid = 'ERAINT_lores_grid'
            resolution = 2.5
        mongo_host_local = 'mongodb://localhost:27017/'
        mg = pymongo.MongoClient(mongo_host_local)
        db = mg.ECMWF
        return({'con_grid': db[col_grid], 'con_anom': db[col_anom]})

    def _queryAnom(self, variable, grid_df):
        ''' Query anomalies for a given variable and for
        a set of grid cells '''

        con_anom = self.con_anom
        grid_ids = grid_df.id_grid.values
        res = con_anom.aggregate(pipeline=[
            {"$project": {"id_grid": 1,
                          "date": 1,
                          variable: 1,
                          "month": {"$month": "$date"}}},
            {"$match": {"month": {"$in": [9, 10, 11, 12, 1, 2]},
                        "id_grid": {"$in": grid_ids.tolist()}}},
            {"$project": {"_id": 0, "id_grid": 1, "date": 1, variable: 1}}
        ])
        anom_df = pd.DataFrame(list(res))
        return anom_df

    def _getPCAscores(self, variable, grid_df):
        ''' 
        Get the PCA scores for a given variable and for
        a set of grid cells.
        This is the former "queryScores" function.
        '''
        anom_df = Predictor._queryAnom(variable, grid_df)
        X_df = anom_df.pivot(index='date',
                             columns='id_grid',
                             values=variable)
        pca = PCA(n_components=3)
        df_scores = pd.DataFrame(pca.fit_transform(X_df),
                                 columns=['PC1_%s' % (variable),
                                          'PC2_%s' % (variable),
                                          'PC3_%s' % (variable)],
                                 index=X_df.index)
        return df_scores

    @staticmethod
    def _genCircle(start_lon, stop_lon, lat, decreasing):
        ''' Helper function for the _createQueryAboveLat function.
        In order to create a straight line on a sphere, it is necessary to 
        generate many points along a given latitude and to connect them.'''
        res = map(lambda x:[int(x), lat],
                  sorted(np.arange(start=start_lon, stop=stop_lon+1),
                         reverse=decreasing))
        return list(res)


    
    @staticmethod
    def _createQueryAboveLat(aboveLat):
        ''' Create geoQuery for the _getGridIds function.'''
        this_box = {'lonmin': -180,
                    'lonmax': 180,
                    'latmin': aboveLat,
                    'latmax': 90}
        circle_north_pos = Predictor._genCircle(start_lon=this_box['lonmin'],
                                                stop_lon=this_box['lonmax'],
                                                lat=this_box['latmax'],
                                                decreasing=False)
        circle_south_neg = Predictor._genCircle(start_lon=this_box['lonmin'],
                                                stop_lon=this_box['lonmax'],
                                                lat=this_box['latmin'],
                                                decreasing=True)
        slp_poly = [[this_box['lonmin'], this_box['latmin']]]
        slp_poly.extend(circle_north_pos)
        slp_poly.extend(circle_south_neg)
        this_polygon = slp_poly

        if aboveLat > 0:
            geo_qry = {"loc":
                       {"$geoWithin": {
                        "$geometry": {
                        "type": "Polygon",
                        "coordinates": [this_polygon]
                        }}}}
        else:  # case of a big polygon larger than one hemisphere
            geo_qry = {"loc":
                   {"$geoWithin": {
                       "$geometry": {
                           "type": "Polygon",
                           "coordinates": [list(reversed(this_polygon))],  # the orientation matters
                           "crs": {
                               "type": "name",
                               "properties": {"name": "urn:x-mongodb:crs:strictwinding:EPSG:4326" }
                           }
                       }
                   }}}
        return(geo_qry)
    
    def _getGridIds(self, aboveLat=None, polygon=None):
        ''' Get the set of all ERAINT grid cells either
        (i) above a given latitude or (ii) within a given polygon '''
        # This funtion replaces: queryGrids(above) and getGridIds(polygon)
        con_grid = self.con_grid

        if (aboveLat is not None):
            geo_qry = Predictor._createQueryAboveLat(aboveLat=aboveLat)
        elif (polygon is not None):
            geo_qry = {"loc":
                       {"$geoWithin": {
                           "$geometry": {
                               "type": "Polygon",
                               "coordinates": polygon
                   }
               }}}
            
        res = con_grid.find(filter=geo_qry, projection={"_id": 0,
                                                        "id_grid": 1,
                                                        "loc": 1})
        grid_df = pd.DataFrame(list(res))
        return grid_df

    @staticmethod
    def _setWinterYear(date):
        ''' December belongs to next year's winter '''
        mon = date.month
        yr = date.year
        if mon >= 9:
            res = yr+1
        else:
            res = yr
        return res
    
    @staticmethod
    def _assignWyear(df):
        ''' Create a new column wyear corresponding to the winter year '''
        res_df = df.assign(
            year=list(map(lambda x: x.year, df.date)),
            wyear=list(map(lambda x: Predictor._setWinterYear(x), df.date)),
            month=list(map(lambda x: x.month, df.date)))
        return res_df

    @staticmethod
    def _renCol(x, mon):
        '''
        Create the Predictor DataFrame
        '''
        if ('PC' in x or 'Nino' in x):
            z = '%s_%s' % (x, mon)
        else:
            z = x
        return z
    
    @staticmethod
    def createMondf(this_mon, scores_df):
        mon_df = scores_df.query('month == @this_mon')
        mon_df.columns = list(
            map(lambda x: Predictor._renCol(x, mon=this_mon),
                list(mon_df)))
        mon_df = mon_df.drop(['date', 'year', 'month'], axis=1)
        return mon_df

    def getPredictorsDF(self):
        ''' This is the main function '''

        # Get grid cell ids above 20°N and 20°S
        grid_df_20N = Predictor._getGridIds(aboveLat=20)
        grid_df_20S = Predictor._getGridIds(aboveLat=-20)

        # Get grid cell ids within the Northern Atlantic region.
        poly_NAtlantic = [list(reversed(
            [[-100, 0], [-100, 45], [-100, 89], [-40, 89],
             [20, 89], [20, 45], [20, 0], [-40, 0], [-100, 0]]))]
        grid_df_NAtlantic = Predictor._getGridIds(polygon=poly_NAtlantic)

        # Work with the Nino Index:
        # No need to compute PCA as Nino index is already
        # a sythetic value. Take it "as is".
        # Region to retrieve Niño 3.4 index for SST
        # Niño 3.4 region: stretches from the 120th to 170th meridians
        # west longitude astride 
        # the equator five degrees of latitude on either side (Wikipedia)
        poly_Nino = [list(reversed([ [-170,-5], [-170,5],[-120,5],
                                     [-120,-5], [-170,-5]]))]
        grid_df_Nino = Predictor._getGridIds(polygon=poly_Nino)
        anom_sst_df = Predictor._queryAnom(variable='sst',
                                           grid_df=grid_df_Nino)
        nino_df0 = anom_sst_df[['date', 'sst']].groupby('date').mean().\
                   reset_index().rename(columns={'sst': 'Nino'})
        
        # Get the PCA scores
        scores_z70 = Predictor._getPCAscores(variable='z70',
                                             grid_df=grid_df_20N)
        scores_ci = Predictor._getPCAscores(variable='ci',
                                            grid_df=grid_df_20N)
        scores_sst = Predictor._getPCAscores(variable='sst',
                                             grid_df=grid_df_20S)
        scores_sst_NAtl = Predictor._getPCAscores(variable='sst',
                                                  grid_df=grid_df_NAtlantic)
        scores_sst_NAtl = scores_sst_NAtl.rename(
            columns={"PC1_sst": "PC1_sstna",
                     "PC2_sst": "PC2_sstna",
                     "PC3_sst": "PC3_sstna"})
        

        # Group all predictors in one DataFrame
        scores_df = pd.merge(left=scores_z70,
                             right=scores_ci,
                             left_index=True,
                             right_index=True).\
                    pipe(lambda df: pd.merge(df,
                                             scores_sst,
                                             left_index=True,
                                             right_index=True)).\
                    pipe(lambda df: pd.merge(df,
                                             scores_sst_NAtl,
                                             left_index=True,
                                             right_index=True))
        scores_df.reset_index(level=0, inplace=True)
        scores_df0 = Predictor._assignWyear(df=scores_df)
        nino_df = Predictor._assignWyear(df=nino_df0)
        scores_df = pd.merge(scores_df0, nino_df)

        # Create the Predictor DataFrame based on the months of
        # November and December:
        sep_df = Predictor._createMondf(this_mon=9, scores_df=scores_df)
        oct_df = Predictor._createMondf(this_mon=10, scores_df=scores_df)
        X_df = pd.merge(sep_df, oct_df)

        # Before applying the Lasso, it is necessary
        # to standardize the predictor ("scander" stuff here)
        # (...)
        # HERE !!!!!!!!!!!!!!!!
        self.X_df = X_df


class StationPrediction():
    ''' Performs predictions for a single station. The output is
    a prediction for the average temperature value over the target months '''
    
    def __init__(self, station_id, target_months, X_df):
        self.station_id = station_id  # GHCN id
        self.target_months = target_months
        self.station_name = Null  # Human-readable station name
        self.X_df = X_df  # Predictor DataFrame
        self.data_df = pd.DataFrame()  # aggregated data over the target months
        self.anom_df = pd.DataFrame()  # station anomalies to be predicted
        self.fit  # the 'fit' object itself
        self.nyears_used  # number of years used for the fit
        self.importance_df = pd.DataFrame()  # what are the dominant pred var?
        self.R2 = Null  # performance of the fit expressed as R2
        self.predictedAnomaly = Null  # the final prediction for the unobserved year
        
    def queryData(self, mon):
        ''' Query station data for an array of months.
        The final result is an *single* averaged value over
        the selected months, for each station.
        E.g. : queryData(mon=['12', '1']) for December and January data.'''
        
        station_id = self.station_id

        self.data_df = data_df
        pass

    def getAnomalies(self):
        ''' The anomalies stands in the foreground, not the bare data.
        Hence, we calculate the yearly anomalies from long-term mean.'''
        data_df = self.data_df

        # ...
        self.anom_df = anom_df
        pass
        
    def fitAnomalies(self, X_df):
        anom_df = self.anom_df
        # ...do the Lasso regression ...

        self.fit = fit  # the entire 'fit' object
        self.R2 = R2
        self.importance_df = importance_df
        self.nyears_used = nyears_used
        pass

    def predictFutureAnomalies(self, newX_df):
        # newX_df are the *new* predictor values
        fit = self.fit
        predictedAnomaly = fit(newX_df)  # something like that
        self.predictedAnomaly = predictedAnomaly
        
        
    
    
    
