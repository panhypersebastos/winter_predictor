# Creation of the reference grid collection for ERA-interim dataset
# Grid collection name : "_grid"
# last modified by: dmasson
# last modified date: 2017-11-12

# python /home/dmasson/CloudStation/code/winter_predictor/era-interim_grid.py > /home/dmasson/data/logfiles/era-interim_grid.log & 

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid
import pymongo
from pprint import pprint
from datetime import datetime, timedelta, date

ERA_vers = 'lores' # or 'hires'

downloadDir = '/home/dmasson/data/era-interim/%s/' % (ERA_vers)

if (ERA_vers == 'hires'):
    col_dat = 'ERAINT_monthly'
    col_grid = 'ERAINT_grid'
    resolution = 0.25
elif (ERA_vers == 'lores'):
    col_dat = 'ERAINT_lores_monthly'
    col_grid = 'ERAINT_lores_grid'
    resolution = 2.5

# Open any era_int file
nc_file = '%sera-int_multivarm1_%s_1979-01-01_to_2017-08-31.nc' % (downloadDir, ERA_vers)


fh = Dataset(nc_file, mode='r')

lons = fh.variables['longitude'][:]
lats = fh.variables['latitude'][:]
t2m = fh.variables['t2m'][:]
fh.close()

# Shift the grid so lons go from -180 to 180 instead of 0 to 360.
t2m_shift, lons_shift = shiftgrid(
    lon0=180., datain=t2m, lonsin=lons, start=False)

lon, lat = np.meshgrid(lons_shift, lats)
this_field = t2m_shift[11, :, :]

# Insert into the grid collection:
mongo_host_local = 'mongodb://localhost:27017/'
con = pymongo.MongoClient(mongo_host_local)
db = con.ECMWF

this_id = 0
for (i, j), val in np.ndenumerate(this_field):
    this_id += 1
    db[col_grid].insert_one({
        "id_grid": this_id,
        "loc": {"type": "Point",
                "coordinates": [float(lon[i, j]), float(lat[i, j])]}
    })

# Create geospatial indexes:
# Warning: geospatial index require -180, +180 longitudes !!
db[col_grid].create_index(
    [("loc", pymongo.GEOSPHERE), ("id_grid", pymongo.ASCENDING)])

endTime = datetime.now()
print("%s %s:%s Job Done !!!" % (endTime.date(), endTime.hour, endTime.minute))
