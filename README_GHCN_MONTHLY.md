# GHCN MONTHLY FEED

**Goal**: Load and update the GHCN station data at the monthly time scale.

The station data is sourced and ingested into MongoDB using the code in `scripts/01_ghcn_monthly_feed.py`. It updates the station data to the latest state.

## DATA SOURCE

* The data is obtained via the [GHCN](https://www1.ncdc.noaa.gov/pub/data/ghcn/v4/) portal.
* All the details from the official and original data description is in [data/readme_ghcn_monthly_official.md](data/readme_ghcn_monthly_official.md).
* The data file has the following name pattern: `ghcnm.VAR.VERSION.qcu.dat`
* There are 12 columns, one for each of the 12 months.
* The *station loccation* file has the following name pattern: `ghcnm.VAR.VERSION.qcu.inv`
* The [metadata](https://www1.ncdc.noaa.gov/pub/data/ghcn/v4/readme.txt) indicates that the *value* column is of type INTEGER with:
    * PRCP = Precipitation (tenths of mm).
    * TMAX = Maximum temperature (tenths of degrees C).
    * TMIN = Minimum temperature (tenths of degrees C).
