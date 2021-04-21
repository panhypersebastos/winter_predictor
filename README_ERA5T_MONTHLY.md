# ERA5T DATA


Let’s start by downloading and exploring the monthly **ERA5T** datasat:

* era_interim_download_monthly.py : script to download ERA-interim NetCDF files.
* era-interim_exploration.ipynb : get familiar with the content of an ERA-interim NetCDF file.

Then, let’s ingest ERA-interim data into MongoDB.

Code:

* era-interim_grid.py : _creation of the grid collection._
* era_interim_insert.py : _creation of the data collection._
* era-interim_exploration.ipynb : _get familiar with the newly created ERA-interim collections._


* The goal is to download meteorological re-analysis data from the European Centre for Medium-Range Weather Forecasts ([ECMWF](https://www.ecmwf.int/)) [ERA5](https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation) dataset.
* Among a vast plethora of possible variables (see [here](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) for example and [here]() for the variable descriptions), we decided to source the following variables:
    * TO BE ADAPTED TO WINTER PREDICTOR !!!
    * **t2m**: air surface temperature at 2m height.
    * **tp**: total precipitation (in meters/hour).
    * **tcc**: total cloud cover (this parameter is the proportion of a grid box covered by cloud. Cloud fractions vary from 0 to 1).
    * **fg10**: wind gust at 10m height

The loaded spatial resolution is 0.25°x0.25° (i.e. roughly **30km horizontal resolution**) and the loaded temporal resolution is **hourly**.

### General Data description

This describes what ERA5T can potentially offer:

* Data type:	gridded
* Horizontal coverage: global
* Horizontal resolution: reanalysis: 0.25°x0.25° (atmosphere), 0.5°x0.5° (ocean waves), Mean, spread and members: 0.5°x0.5° (atmosphere), 1°x1° (ocean waves)
* Temporal coverage:	1979 to present day !!! ADAPT !!!
* Temporal resolution: hourly
* File format: [netCDF](https://www.unidata.ucar.edu/software/netcdf/)
* Update frequency: daily
* Available variables: see [this table](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview)

### Where is the data stored ?
TO BE ADAPTED TO MONGODB

* Database: `postgres`
* Grid table: `era5t.era5t_grid`. Additional indexes on: id_grid (asc)
* Data table: `era5t.era5t_data`. Additional indexes on: time (desc), id_grid (asc)

**Indexes**: as soon as posssible, create the additional indexes specified above.

### Data update frequency

Original information available [here](https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation). Initial release data, i.e. data no more than three months behind real time, is called **ERA5T**.

* For the **CDS**, daily updates are available **5 days behind real time** and monthly mean updates are available 5 days after the end of the month.
* For netCDF data, there is no means of differentiating between ERA5 and ERA5T data.


## ECWMF's CDS  API
You need to use ECMWF's API in order to download ERA5 data. Read [CDS API documentations](https://cds.climate.copernicus.eu/api-how-to) for more details.


###Install cdsapi
See conda environment below, or:
```bash
pip install cdsapi 
```

Your $HOME/.cdsapirc file shall contain:
```bash
url: https://cds.climate.copernicus.eu/api/v2
key: <your key>
```

### CDS commands & FAQ
- Go [here](https://cds-dev.copernicus-climate.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form), select the desired data and then click on *show API request*.

- Some code example to retrieve [daily data](https://confluence.ecmwf.int/display/WEBAPI/ERA5+daily+retrieval+efficiency)

- Does a variable belong to analysis or forecast? See link [here](https://confluence.ecmwf.int/display/CKB/ERA5+data+documentation).
