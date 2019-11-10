# GHCN MONTHLY FEED


## DATA SOURCE

* [GHCN](ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/v4/) portal
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