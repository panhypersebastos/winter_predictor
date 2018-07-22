# This file aims to contain all important code
# in the object-oriented module format

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
    with month1 and month2 being November and December generally '''

    def __init__(self):
        self.con_anom = None,  # ERAINT anomalies collection
        self.con_grid = None,  # ERAINT grid collection
        self.anom_df = pd.DataFrame() # anomalies for a set of grid cells
        self.X_df = pd.DataFrame() # final DataFrame of predictors

    def initializeCon(self, ERA_vers='lores'):
        ''' Create the connections to the MongoDB collections '''
        pass

    @staticmethod
    def _queryAnom(variable, grid_df):
        ''' Query anomalies for a given variable and for 
        a set of grid cells '''
        # self.anom_df = anom_df
        pass

    def _getPCAscores(self, variable, grid_df):
        ''' blabla '''
        # Former queryScores function
        anom_df = Predictor._queryAnom(variable, grid_df)
        pass

    def _getGridIds(self, aboveLat=None, polygon=None):
        ''' Get the set of all ERAINT grid cells either 
        (i) above a given latitude or (ii) within a given polygon '''
        # This funtion replaces: queryGrids(above) and getGridIds(polygon) 
        con_grid = self.con_grid
        pass

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

    
    def getPredictorsDF(self):
        ''' This is the main function '''
        
        # Get grid cell ids above 20°N and 20°S 
        grid_df_20N = Predictor._getGridIds(aboveLat=20)
        grid_df_20S = Predictor._getGridIds(aboveLat=-20)

        # Get grid cell ids within the Northern Atlantic region
        grid_df_NAtlantic = Predictor._getGridIds(polygon=.....)

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
        scores_df0 = assignWyear(df=scores_df)
        nino_df = assignWyear(df=nino_df0)
        scores_df = pd.merge(scores_df0, nino_df)

        # Create the Predictor DataFrame based on the months of
        # November and December:
        sep_df = createMondf(this_mon=9, scores_df=scores_df)
        oct_df = createMondf(this_mon=10, scores_df=scores_df)
        X_df = pd.merge(sep_df, oct_df)
        self.X_df = X_df

class StationPrediction():
    ''' Performs predictions for a single station. The output is 
    a prediction for the average temperature value over the target months '''
    
    def __init__(self, station_id, target_months, X_df):
        self.station_id = station_id # GHCN id
        self.target_months = target_months
        self.station_name = Null # Human-readable station name
        self.X_df = X_df # Predictor DataFrame
        self.data_df = pd.DataFrame() # station data for the target months
        

    def queryData(self, mon):
        ''' Query station data for an array of months.
        E.g. : queryData(mon=['12', '1']) for December and January data '''
        station_id = self.station_id

        self.data_df = data_df
        pass

    def getAnomalies(self):
        
    
