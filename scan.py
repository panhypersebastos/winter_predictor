'''
WINTER PREDICTOR SCAN

This code:

* scans all stations in selected countries
* fit a prelim regression on z90 and sea-ice in sept and oct
* if the fit is not too bad, a lasso regularization is applied
* if the R2 is high, if enough years are available,
  and if the anomaly is remarkable THEN:
* enters the next bet in the winter_pred database
Uses the "winter_predictor" oriented-object modules.
'''

from winter_predictor import Predictor, StationPrediction
import numpy as np
import pymongo
import pandas as pd
from scipy.stats import norm
import logging
import os
from multiprocessing import Pool
import psutil

# YOUR INPUT HERE :
# (i) Target Months:
target_months = ['12', '1']
# (ii) Predictor month(s)
predi = 'aug'

# Physical cores
num_cores = psutil.cpu_count(logical=False)

# Logiging
logfilename = '/home/dmasson/CloudStation/code/winter_preditor/scan_%s.log' % predi
if os.path.exists(logfilename):
        os.remove(logfilename)
logging.basicConfig(filename=logfilename,
                    format='%(asctime)s %(message)s', level=logging.INFO)
logging.info('WINTER PREDICTOR SCAN: Predicting %s' % predi)

'''
PART 1 : GET PREDICTORS
The **SAME** predictor set will then be used
for any target station to predict.
'''

PRED = Predictor()
PRED.getPredictorsDF(predi=predi)

# Gather all relelvant station ids
mongo_host_local = 'mongodb://localhost:27017/'
mg = pymongo.MongoClient(mongo_host_local)
db = mg.GHCN
countries = pd.read_csv('input.csv').name.values
sta_df = pd.DataFrame(list(
    db.stations.find(filter={
        'country': {'$in': list(countries)}})))
all_ids = list(sta_df['station_id'].values)
logging.info('The following countries will be analyzed: %s' % countries)
logging.info('Total number of stations: %s' % len(all_ids))


def scanStation(id,
                wyear_new,
                min_r2_prelim,
                min_r2,
                min_nyears):
        print(id)
        STA = StationPrediction(station_id=int(id),
                                target_months=target_months,
                                X_df=PRED.X_df)
        STA.queryData()
        STA.getAnomalies()
        # PART 3 : FIT ANOMALIES
        # Quick fit first
        STA.quickfitAnomalies(X_df=PRED.X_df)
        STA.R2_prelim, STA.nyears_used

        if STA.R2_prelim > min_r2_prelim and STA.nyears_used > min_nyears:
                # Do Lasso Regression
                STA.fitAnomalies(X_df=PRED.X_df)
                if STA.R2 > min_r2:
                        logging.info('Interesting case in %s for station %s with R2=%s' %
                                     (STA.metadata['country'],
                                      STA.metadata['name'],
                                      STA.R2))
                        # PART 4 : PREDICT FUTURE ANOMALIES
                        newX_df = PRED.X_df.query('wyear==@wyear_new')
                        STA.predictFutureAnomalies(newX_df)
                        pred_anomaly = STA.predictedAnomaly

                        # PART 5 : IS EXTREME ANOMALY ?
                        # Long-term trend
                        ltt = STA.detrend_fit
                        ltt_fit = ltt.predict(wyear_new)
                        # Predicted wyear value :
                        T_pred = pred_anomaly + ltt_fit
                        # Anomaly as defined by SwissRe :
                        SwissRe_df = STA.anom_df.tail(10)  # take only the last 10 years
                        SwissRe_ltt = np.nanmean(SwissRe_df.x.values)
                        SwissRe_anom = T_pred-SwissRe_ltt
                        SwissRe_df = SwissRe_df.assign(anom_SwissRe=SwissRe_df.x-SwissRe_ltt)
                        SwissRe_df = SwissRe_df.dropna()
                        # Fit Normal distribution
                        mu, std = norm.fit(SwissRe_df.anom_SwissRe)
                        quantl = norm.cdf(0, loc=SwissRe_anom, scale=std).ravel()[0]
                        
                        # If extreme, then store the result
                        if quantl > 0.75 or quantl < 0.25:
                                logging.info('Extreme anomaly confirmed in %s for station %s' %
                                             (STA.metadata['country'],
                                              STA.metadata['name']))
                                # Summary document:
                                pred_doc = {'wyear': wyear_new,
                                            'station_id': STA.metadata['station_id'],
                                            'country': STA.metadata['country'],
                                            'name': STA.metadata['name'],
                                            'R2': STA.R2,
                                            'nyears_used': STA.nyears_used,
                                            'SwissRe_anom': SwissRe_anom.ravel()[0],
                                            'pred_T': T_pred.ravel()[0],
                                            'quantile_anom': quantl}
                                mongo_host_local = 'mongodb://localhost:27017/'
                                mg = pymongo.MongoClient(mongo_host_local)
                                db = mg.winter_pred
                                db.prediction.insert_one(pred_doc)


def scanStation_wrap(id):
        try:
                scanStation(id=id,
                            wyear_new=2017,
                            min_r2_prelim=0.4,
                            min_r2=0.5,
                            min_nyears=20)
        except ValueError:
                print('error here')


void = list(map(scanStation_wrap, all_ids))

pred_col = mg.winter_pred.prediction
ndocs = pred_col.count()
logging.info('JOB DONE, %s cases found' % ndocs)
