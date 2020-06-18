# ERA5T DATA

- List of ERA5 variables [here](https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation).
- More details on variables [here](https://apps.ecmwf.int/codes/grib/param-db).

### Data update frequency
Original information available [here](https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation). Initial release data, i.e. data no more than three months behind real time, is called **ERA5T**.  In the event that serious flaws are detected in ERA5T, this data could be different to the final ERA5 data. In practice, though, this will be very unlikely to occur. Based on experience with the production of ERA5 so far (and ERA-Interim in the past), our expectation is that such an event would not occur more than once every few years, if at all. In the unlikely event that such a correction is required, users will be notified as soon as possible.

* For the **CDS**, daily updates are available **5 days behind real time** and monthly mean updates are available 5 days after the end of the month.
* For MARS ERA5 data, monthly updates are available about two months after the month in question. In the future, ERA5T data will also be available in MARS.
* For GRIB data, ERA5T can be identified by the key expver=0005 in the GRIB header. ERA5 is identified by the key expver=0001.
* For netCDF data, there is no means of differentiating between ERA5 and ERA5T data.


## ECWMF's CDS  API
You need to use ECMWF's API in order to download ERA5 data. Read [CDS API documentations](https://cds.climate.copernicus.eu/api-how-to) for more details.


###Install cdsapi
See conda environment below, or:
```bash
pip install cdsapi 
```

The .cdsapirc file contains:
```bash
url: https://cds.climate.copernicus.eu/api/v2
key: 14047:627b341d-a1ac-46f4-97e1-7849371cc2a4
```

### CDS commands
- Go [here](https://cds-dev.copernicus-climate.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form), then *show API request*.

- Get [daily data](https://confluence.ecmwf.int/display/WEBAPI/ERA5+daily+retrieval+efficiency)

- Does a variable belong to analysis or forecast? See link [here](https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation).


## CONDA ENVIRONMENT

* For the python option
* Name: era5t
* Activation: conda activate era5t
* Packages installation:

```code
conda create -n era5t python=3
conda activate era5t

conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

conda install xarray dask netCDF4 bottleneck
conda install pymssql
pip install git+https://bitbucket.com/celsiuspro/pycputils
conda install -c mvdbeek multiprocessing-logging
conda install joblib
conda install jupyter
conda install cdsapi
conda install matplotlib

# the packages below are not installed (and are not strictly necessary)
conda install cdo
conda install python-cdo
conda install geopandas
conda install descartes
conda install -c oggm salem
conda install rasterio

```