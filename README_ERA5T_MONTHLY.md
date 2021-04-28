# ERA5T DATA

## General Data description

ERA5T is sourced in the following setup:

| Topic | Description|
|:-------------|:-------------|
|Data type|gridded|
|Horizontal coverage| global|
|Horizontal resolution| 0.25°x0.25° (i.e. ~30x30km²)|
|Temporal coverage|	1979 to present day|
|Temporal resolution| monthly|
|File format| [netCDF](https://www.unidata.ucar.edu/software/netcdf/)|
|Update frequency| daily, with a lag of ~5 days, see paragraph below|
|Available variables| see [this table](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)|

Among a plethora of available variables (see [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) for an example and [here](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation) for the variable descriptions), following variables were stored:


| Abbreviation | Variable name|
|:-------------|:-------------|
| ci | [Sea-ice cover](https://apps.ecmwf.int/codes/grib/param-db?id=31) (0-1)|
| sp | [Surface pressure](https://apps.ecmwf.int/codes/grib/param-db?id=134) (Pa) |
| sst | [Sea surface temperature](https://apps.ecmwf.int/codes/grib/param-db?id=34) (K) |
| z70 | [Geopotential height at 70 hPa height](https://apps.ecmwf.int/codes/grib/param-db?id=129) (m²/s²)|


### More about the update frequency:
The original data information is available [here](https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation). The initial released data with no more than three months behind real time, is called **ERA5T**. For the **CDS**, daily updates are available **5 days behind real time** and monthly mean updates are available 5 days after the end of the month. For netCDF data, there is no means of differentiating between ERA5 and ERA5T data.

## ECWMF's CDS  API
You need to use cdsapi (i.e., ECMWF's API) in order to download ERA5 data. Read [CDS API documentations](https://cds.climate.copernicus.eu/api-how-to) for more details. In short, a (free) registration to Copernicus and the setting up of ECMWF's API are required. Both steps are easily done, see the sketch below:


### Installation of cdsapi
If not already done in the virtual environment, execute `pip install cdsapi`


Your `$HOME/.cdsapirc` file shall contain:
```bash
url: https://cds.climate.copernicus.eu/api/v2
key: <your key>
```
### CDS commands & FAQ
If you need help on the API on how to source specific data, go [here](https://cds-dev.copernicus-climate.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form), select the desired data and then click on *show API request*. Here are some code examples to retrieve [daily data](https://confluence.ecmwf.int/display/WEBAPI/ERA5+daily+retrieval+efficiency). Whether a variable belongs to analysis or forecast is answered [here](https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation).

## To be deleted


Let’s start by downloading and exploring the monthly **ERA5T** datasat:

* era_interim_download_monthly.py : script to download ERA-interim NetCDF files.
* era-interim_exploration.ipynb : get familiar with the content of an ERA-interim NetCDF file.

Then, let’s ingest ERA-interim data into MongoDB.

Code:

* era-interim_grid.py : _creation of the grid collection._
* era_interim_insert.py : _creation of the data collection._
* era-interim_exploration.ipynb : _get familiar with the newly created ERA-interim collections._


* The goal is to download meteorological re-analysis data from the European Centre for Medium-Range Weather Forecasts ([ECMWF](https://www.ecmwf.int/)) [ERA5](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation) dataset.
* 





