# GHCN MONTHLY FEED

Loads the GHCN station data at the monthly time scale.

* The feed is in `pred/ghcn_monthly_feed.py`. It updates the station data.
* The class can be tested in `dev/ghcn_monthly_class_test.ipynb`
* The notebook `dev/ghcn_monthly_data.ipynb` once served as a preliminary sketch of the feed


Collection created in MongoDB:

* Database name: __GHCNM__
* Station collection: __stations__

## DATA SOURCE

* [GHCN](https://www1.ncdc.noaa.gov/pub/data/ghcn/v4/) portal
* The **data** file has the following name pattern: ghcnm.VAR.VERSION.qcu.dat
* There are 12 columns, one for each of the 12 months
* The *station colcation* file has the following name pattern: ghcnm.VAR.VERSION.qcu.inv
* The [metadata](https://www1.ncdc.noaa.gov/pub/data/ghcn/v4/readme.txt) indicates that the *value* column is of type INTEGER with:
* * PRCP = Precipitation (tenths of mm)
* * TMAX = Maximum temperature (tenths of degrees C)
* * TMIN = Minimum temperature (tenths of degrees C)
