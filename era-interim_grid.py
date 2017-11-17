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

# Open any era_int file
nc_file = '/home/dmasson/data/era-interim/era-int_multivarm1_1979-01-01_to_2017-08-31.nc'
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
    db.ERAINT_grid.insert_one({
        "id_grid": this_id,
        "loc": {"type": "Point",
                "coordinates": [float(lon[i, j]), float(lat[i, j])]}
    })

# Create geospatial indexes:
# Warning: geospatial index require -180, +180 longitudes !!
db.ERAINT_grid.create_index(
    [("loc", pymongo.GEOSPHERE), ("id_grid", pymongo.ASCENDING)])

endTime = datetime.now()
print("%s %s:%s Job Done !!!" % (endTime.date(), endTime.hour, endTime.minute))
