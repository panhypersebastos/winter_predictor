# GHCN MONTHLY FEED

Loads the GHCN station data at the monthly time scale.

* Check the notebook __GHCN/ghcn_monthly_data.ipynb__ for a preliminary sketch of the feed
* The feed itself is __CHCN/ghcn_monthly_feed.py__
* The class is being tested in __GHCN/ghcn_monthly_class_test.ipynb__

Collection created in MongoDB:

* Database name: __GHCNM__
* Station collection: __stations__

## DATA SOURCE

* [GHCN](ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/v4/) portal
* The **data** file has the following name pattern: ghcnm.VAR.VERSION.qcu.dat
* There are 12 columns, one for each of the 12 months
* The *station colcation* file has the following name pattern: ghcnm.VAR.VERSION.qcu.inv
* The [metadata](ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/v4/readme.txt) indicates that the *value* column is of type INTEGER with:
* * PRCP = Precipitation (tenths of mm)
* * TMAX = Maximum temperature (tenths of degrees C)
* * TMIN = Minimum temperature (tenths of degrees C)


## CONDA ENVIRONMENT


```code
conda create -n ghcn
conda activate ghcn

conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

conda install pymssql
pip install git+https://bitbucket.com/celsiuspro/pycputils
conda install -c mvdbeek multiprocessing-logging
conda install joblib
conda install jupyter
conda install matplotlib
conda install flake8
conda install pyspark
# conda install ipdb


```