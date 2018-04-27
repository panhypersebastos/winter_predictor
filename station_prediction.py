
# coding: utf-8

# ## Station Prediction
# 
# This code bring two parts together:
# * station_exploration.ipynb
# * era-int_NAO_prediction.ipynb

# In[1]:


from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import pymongo
from pprint import pprint
from datetime import datetime, timedelta, date
import pandas as pd
from sklearn.decomposition import PCA
import sklearn.linear_model as skl_lm
import gdal as gdl
import matplotlib.mlab as ml
import cartopy.crs as ccrs
import plotly.graph_objs as go
import plotly.offline as py
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC

py.init_notebook_mode(connected=True) # for live plot
pd.set_option('display.notebook_repr_html', False)
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn-white')


# In[2]:


mongo_host_local = 'mongodb://localhost:27017/'
mg = pymongo.MongoClient(mongo_host_local)


# In[3]:


db = mg.ECMWF
db.collection_names()


# In[4]:


ERA_vers = 'lores'
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

con_grid = db[col_grid]
con_anom = db[col_anom]


# ## Name of variables:
# 
# * 'ci':  Sea-ice cover [0-1]
# * 'sst': Sea surface temperature [K]
# * 'istl1': Ice temp layer1 [K]
# * 'sp': Surface pressure [Pa]
# * 'stl1': Soil temp lev1 [K]
# * 'msl': Mean SLP [Pa]
# * 'u10': wind-u [m/s]
# * 'v10': 
# * 't2m': 2m temp [K]
# * 'd2m': 2m dewpoint temp.[K]
# * 'al': Surface albedo [0-1]
# * 'lcc': Low cloud cover [0-1]
# * 'mcc': Medium cloud cover [0-1]
# * 'hcc': High cloud cover [0-1]
# * 'si10': 10m wind speed [m/s]
# * 'skt': Skin temperature [K]
# * 'blh': Boundary layer hgt [m]
# * 'ishf': Inst.surf.sensbl.heatflux [W/m2]
# * 'ie': Instantaneous moisture flux [kg*m^-2*s^-1]
# * 'z70': Geopot. height @70hPa [m]

# In[5]:


# Names of candidate variables:
fo0 = con_anom.find({}, {'_id': 0, 'year': 0, 'month': 0, 'date': 0, 'id_grid': 0}).limit(1)
fo_df = pd.DataFrame(list(fo0))
all_varnames = list(fo_df)
all_varnames


# # Get Predictors

# In[6]:


# Query anomalies for a variable for each input grid cells
def queryAnom(this_variable, this_grid_df):
    # Query data anomalies
    grid_ids = this_grid_df.id_grid.values
    res = con_anom.aggregate(pipeline=[ 
    {"$project": {"id_grid": 1, "date": 1, this_variable: 1, "month": {"$month": "$date"}}},
    {"$match": {"month": {"$in": [9, 10, 11, 12, 1, 2]},
                "id_grid": {"$in": grid_ids.tolist()} }},
    {"$project": {"_id": 0, "id_grid": 1, "date": 1, this_variable: 1}} ])    
    anom_df = pd.DataFrame(list(res))
    return anom_df


# In[7]:


# Query grid cells for NAO calculation
poly1 = [list(reversed([ [-50,25], [-50,55], [10,55],[ 10,25], [-50,25]]))]
poly2 = [list(reversed([ [-40, 55], [-40, 85], [20, 85], [20, 55], [-40, 55]]))]
def getGridIds(this_polygon):
    geo_qry = {"loc": 
               {"$geoWithin": {
                   "$geometry": {
                       "type": "Polygon",
                       "coordinates": this_polygon
                   }
               }}}

    res = con_grid.find(filter = geo_qry, projection = {"_id":0, "id_grid": 1, "loc": 1})
    grid_df = pd.DataFrame(list(res))
    return grid_df
grid_df1 = getGridIds(poly1)
grid_ids1 = grid_df1.id_grid.values
grid_df2 = getGridIds(poly2)
grid_ids2 = grid_df2.id_grid.values


# In[8]:


# Region to retrieve Niño 3.4 index for SST
# Niño 3.4 region: stretches from the 120th to 170th meridians west longitude astride 
# the equator five degrees of latitude on either side (Wikipedia)
poly_Nino = [list(reversed([ [-170,-5], [-170,5],[-120,5], [-120,-5], [-170,-5]]))]
grid_df_Nino = getGridIds(poly_Nino)
grid_ids_Nino = grid_df_Nino.id_grid.values
anom_sst_df = queryAnom(this_variable='sst', this_grid_df=grid_df_Nino)
nino_df0 = anom_sst_df[['date', 'sst']].groupby('date').mean().reset_index().rename(columns={'sst':'Nino'})
nino_df0.head()


# In[9]:


# Generic function to query grid ids above a given latitude
def genCircle(start_lon, stop_lon, lat, decreasing): 
    res = map(lambda x:[int(x), lat],
              sorted(np.arange(start=start_lon, stop=stop_lon+1), reverse=decreasing))
    return list(res)

def queryGrids(aboveLat):
    this_box = {'lonmin': -180, 'lonmax': 180, 'latmin': aboveLat, 'latmax': 90}
    circle_north_pos = genCircle(start_lon = this_box['lonmin'], stop_lon = this_box['lonmax'], 
                                  lat = this_box['latmax'], decreasing = False)
    circle_south_neg = genCircle(start_lon = this_box['lonmin'], stop_lon = this_box['lonmax'], 
                                lat = this_box['latmin'],  decreasing = True)
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
    else: # case of a big polygon larger than one hemisphere
        geo_qry = {"loc": 
               {"$geoWithin": {
                   "$geometry": {
                       "type": "Polygon",
                       "coordinates": [list(reversed(this_polygon))], # the orientation matters
                       "crs": {
                           "type": "name", 
                           "properties": { "name": "urn:x-mongodb:crs:strictwinding:EPSG:4326" }
                       }
                   }
               }}}
        
    res = con_grid.find(filter = geo_qry, projection = {"_id":0, "id_grid": 1, "loc": 1})
    grid_df = pd.DataFrame(list(res))
    return grid_df

grid_df_20N = queryGrids(aboveLat=20)
grid_df_20S = queryGrids(aboveLat=-20)


# # Get PCA scores

# In[10]:


# Generic function to query grid ids above a given latitude
def genCircle(start_lon, stop_lon, lat, decreasing): 
    res = map(lambda x:[int(x), lat],
              sorted(np.arange(start=start_lon, stop=stop_lon+1), reverse=decreasing))
    return list(res)

def queryGrids(aboveLat):
    this_box = {'lonmin': -180, 'lonmax': 180, 'latmin': aboveLat, 'latmax': 90}
    circle_north_pos = genCircle(start_lon = this_box['lonmin'], stop_lon = this_box['lonmax'], 
                                  lat = this_box['latmax'], decreasing = False)
    circle_south_neg = genCircle(start_lon = this_box['lonmin'], stop_lon = this_box['lonmax'], 
                                lat = this_box['latmin'],  decreasing = True)
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
    else: # case of a big polygon larger than one hemisphere
        geo_qry = {"loc": 
               {"$geoWithin": {
                   "$geometry": {
                       "type": "Polygon",
                       "coordinates": [list(reversed(this_polygon))], # the orientation matters
                       "crs": {
                           "type": "name", 
                           "properties": { "name": "urn:x-mongodb:crs:strictwinding:EPSG:4326" }
                       }
                   }
               }}}
        
    res = con_grid.find(filter = geo_qry, projection = {"_id":0, "id_grid": 1, "loc": 1})
    grid_df = pd.DataFrame(list(res))
    return grid_df

grid_df_20N = queryGrids(aboveLat=20)
grid_df_20S = queryGrids(aboveLat=-20)


# In[11]:


# 3rd region for SST in Northern Atlantic, as in Promet (2008).
poly_NAtlantic = [list(reversed(
    [ [-100,0], [-100,45],[-100,89], [-40, 89],[20,89],[20,45],[20,0], [-40,0], [-100,0]]))]
grid_df_NAtlantic = getGridIds(poly_NAtlantic)
grid_ids_NAtlantic = grid_df_NAtlantic.id_grid.values


# In[12]:


def queryScores(this_variable, this_grid_df):
    # Query data anomalies
    anom_df = queryAnom(this_variable, this_grid_df)
    # Get Principal Component Scores
    X_df = anom_df.pivot(index='date', columns='id_grid', values=this_variable)
    pca = PCA(n_components=3)
    df_scores = pd.DataFrame(pca.fit_transform(X_df), 
                             columns=['PC1_%s' % (this_variable), 
                                      'PC2_%s' % (this_variable), 
                                      'PC3_%s' % (this_variable)],
                             index=X_df.index)
    return df_scores

scores_z70 = queryScores(this_variable='z70', this_grid_df=grid_df_20N)
scores_ci = queryScores(this_variable='ci', this_grid_df=grid_df_20N)
scores_sst = queryScores(this_variable='sst', this_grid_df=grid_df_20S)
scores_sst_NAtl = queryScores(this_variable='sst', this_grid_df=grid_df_NAtlantic)


# In[13]:


scores_sst_NAtl = scores_sst_NAtl.rename(columns={"PC1_sst": "PC1_sstna",
                                                   "PC2_sst": "PC2_sstna",
                                                   "PC3_sst": "PC3_sstna"})


# ### Group all predictors in one DataFrame

# In[14]:


def assignWyear(df):
    res_df = df.assign(
    year=list(map(lambda x: x.year, df.date)),
    wyear=list(map(lambda x: setWinterYear(x), df.date)), 
    month=list(map(lambda x: x.month, df.date)))
    return res_df


# In[15]:


def setWinterYear(date): # December belong to next year's winter
    mon=date.month
    yr=date.year
    if mon >= 9:
        res = yr+1
    else:
        res = yr
    return res


# In[16]:


scores_df = pd.merge(left=scores_z70, right=scores_ci, left_index=True, right_index=True).pipe(lambda df: pd.merge(df, scores_sst, left_index=True, right_index=True)).pipe(lambda df: pd.merge(df, scores_sst_NAtl, left_index=True, right_index=True))
scores_df.reset_index(level=0, inplace=True)
scores_df0 = assignWyear(df=scores_df)
nino_df = assignWyear(df=nino_df0)
scores_df = pd.merge(scores_df0, nino_df)
scores_df.head()


# In[17]:


# Create the Predictor DataFrame
def renCol(x, mon):
    if ('PC' in x or 'Nino' in x):
        z = '%s_%s' % (x, mon)
    else:
        z = x
    return z

def createMondf(this_mon, scores_df):
    mon_df = scores_df.query('month == @this_mon')
    mon_df.columns = list(map(lambda x: renCol(x, mon=this_mon), list(mon_df)))
    mon_df = mon_df.drop(['date','year','month'], axis=1)
    return mon_df

sep_df = createMondf(this_mon=9, scores_df=scores_df)
oct_df = createMondf(this_mon=10, scores_df=scores_df)
X_df = pd.merge(sep_df, oct_df)


# In[18]:


X_df.head()


# # Get Target Variables

# In[19]:


mongo_host_local = 'mongodb://localhost:27017/'
mg = pymongo.MongoClient(mongo_host_local)
db = mg.GHCN
db.collection_names()


# In[20]:


# Find Swiss stations
sta_df = pd.DataFrame(list(db.stations.find(filter={'country': 'SWITZERLAND'})))
sta_df


# ## Query function
# 
# Create a function that returns the "risk_df" as above, and as a function of the following inputs:
# * station id
# * any groupement of dec, jan, feb (for instance, [dec,jan])
# 

# In[21]:


def queryData(station_id, mon):
    dat_df = pd.DataFrame(list(db.data.find(filter={'station_id': station_id}))).    pipe(lambda df: df[['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'year']]).    pipe(lambda df: df.query('year >= 1979'))
    w_df = dat_df[['year', '1', '2', '12']]
    # Reformat data
    dec_df = w_df[['year', '12']]
    dec_df = dec_df.assign(wyear=dec_df.year+1).pipe(lambda df: df[['wyear', '12']])
    jf_df = w_df[['year', '1', '2']].pipe(lambda df: df.rename(columns={'year':'wyear'}))
    winter_df = pd.merge(dec_df, jf_df, on='wyear')
    # Do the aggregation for december-february risk period
    risk_df = winter_df
    risk_df['ave'] = risk_df[mon].apply(func=np.mean, axis=1)
    risk_df = risk_df[['wyear', 'ave']].pipe(lambda df: df.rename(columns={'ave': station_id}))
    return(risk_df)

res_df = queryData(station_id=64606660000, mon=['12','1'])
res_df.head()


# ### Query function for an ensemble of stations

# In[22]:


ids = sta_df.station_id#[:5]


# In[23]:


# Generic function
def getStationAgg(station_ids, mon):
    all_df00 = list(map(lambda x: queryData(station_id=x, mon=mon), ids))
    all_df = reduce(lambda x,y: pd.merge(x,y,on='wyear', how='outer'), all_df00).    pipe(lambda df: df.sort_values('wyear', ascending=True)).    reset_index(drop=True).    pipe(lambda df: df.dropna(axis=1, thresh=20) ) # NA: at least 20 data obs should be non-NA
    return(all_df)

all_df0 = getStationAgg(station_ids=sta_df.station_id, mon=['12','1'])
all_df0.head()


# In[24]:


# Generic rename function
def station_id_to_name(all_df0):
    idf = all_df0.drop(columns='wyear').columns
    sta_df0 = pd.DataFrame(list(db.stations.find(filter={'station_id': {"$in": list(idf)}})))

    nam_df = sta_df0.query('station_id in @idf').pipe(lambda df: df[['station_id', 'name']])
    newnames = dict(nam_df.to_dict('split')['data'])
    all_df = all_df0.rename(columns=newnames)
    return(all_df)

all_df = station_id_to_name(all_df0=all_df0)
all_df.head()


# # Station Anomalies
# 
# For the moment, let's keep it simple: we predict the anomalies from long-term trend.
# 
# As a side note for later, check SwissRe guidelines:
# 
# * P. 47: *"If the trend in the weather index data is against SRCSGMI, apply detrending on calculated weather index values"*
# 
# * i.e., the index itself is detrended, not the original observations!
# 
# 
# * p.49: For HDD call and put options, *NO* detrending is applied. The historical period is (!!) the last 10 years period.
# 
# This means, we can take advantage of the global temperature trend and use the year index as predictor later.

# In[25]:


# Rem: the following copy would be wrong:
# Problem copy var by reference, see the "Python for Data Science" book p.390 
# ind_df = all_df # Instead do:
anom_df = pd.DataFrame(all_df)

colnames = anom_df.drop(columns='wyear').columns

for colname in colnames:
    #colname = colnames[0]
    model = skl_lm.LinearRegression()
    # Handle the NA problem
    reg_df = anom_df[['wyear', colname]].pipe(lambda df: df.dropna())
    X = reg_df.wyear.values.reshape(-1, 1)
    X_pred = anom_df.wyear.values.reshape(-1, 1)
    y = reg_df[[colname]]
    model.fit(X, y)
    lm_pred = model.predict(X_pred)
    anom_df['fit'] = lm_pred
    anom_df[colname] = anom_df[colname] - anom_df['fit']

anom_df = anom_df.drop(columns='fit')
anom_df.head()


# In[26]:


anom_df.plot(x='wyear', figsize=(15,6))


# ### Create Regression DataFrame

# In[27]:


# Create Regression DataFrame
dat_df = pd.merge(anom_df, X_df, on='wyear')
dat_df.info()


# ## Regularization / Lasso Model Selection

# #### Generic function for any station :

# In[40]:


def predOneStation(target, dat_df):
    # target = 'ZURICH (TOWN/'
    # 'target ~ PC1_ci_10 + PC2_z70_10 + PC3_sst_9' # Wang
    predNames = np.array(['PC1_z70_9',
     'PC2_z70_9',
     'PC3_z70_9',
     'PC1_ci_9',
     'PC2_ci_9',
     'PC3_ci_9',
     'PC1_sst_9',
     'PC2_sst_9',
     'PC3_sst_9', 
     'PC1_z70_10',  
     'PC2_z70_10', 
     'PC3_z70_10',
     'PC1_ci_10', 
     'PC2_ci_10',
     'PC3_ci_10',
     'PC1_sst_10',
     'PC2_sst_10',
     'PC3_sst_10',
     'PC1_sstna_10','PC2_sstna_10','PC3_sstna_10',
                          'Nino_9', 'Nino_10'])
    #ipdb.set_trace()
    dat_df = dat_df[dat_df[target].notnull()] # eliminate NA rows
    X = dat_df[predNames].as_matrix()
    # Target Variables:
    y = dat_df[[target]]
    y = np.ravel(y)
    # Before applying the Lasso, it is necessary to standardize the predictor
    scaler = StandardScaler()
    scaler.fit(X)
    X_stan = scaler.transform(X)
    # In order to find the optimal penalty parameter alpha,
    # use Cross-validated Lasso
    #modlcv = LassoLarsIC(criterion='aic')
    modlcv = LassoCV(cv=3, n_alphas=10000,max_iter=10000)
    modlcv.fit(X_stan, y)
    alpha = modlcv.alpha_

    # Name Of the non-null coefficients:
    # 'target ~ PC1_ci_10 + PC2_z70_10 + PC3_sst_9' # Wang
    ind = np.array(list(map(lambda x: float(x)!=0, modlcv.coef_)))
    importance_df = pd.DataFrame({'pred': predNames[ind], 
                                  'coef': modlcv.coef_[ind]})
    importance_df = importance_df.assign(absCoef=np.absolute(importance_df.coef))
    # According to the Lasso, the 3 strongest predictors are:
    # PC1_ci_10, PC2_z70_10, PC3_z70_9 
    importance_df.sort_values('absCoef', ascending=False)
    res = dict({'R2': modlcv.score(X_stan, y),'alpha': alpha, 'importance_df': importance_df})
    return(res)

z = predOneStation(target='PAYERNE', dat_df=dat_df)
z


# In[41]:


z = map(predOneStation, sta_df.name)


# In[56]:


for sta_name in colnames:
    z = predOneStation(target=sta_name, dat_df=dat_df)
    if z['R2'] > 0.4:
        print(sta_name, z)


# In[57]:


dat_df


# In[48]:


colnames

