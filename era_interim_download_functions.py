from datetime import datetime, date
import numpy as np
import itertools
import pandas as pd

def getMultivar(from_day, to_day, downloadDir, server):
    # get MULTIPLE VAIABLES :
    filename_multivar = 'era-int_multivar_%s_to_%s.nc' % (from_day, to_day)
    print("PROCESSING %s..." % (filename_multivar))
    t_interval = "%s/to/%s" % (from_day, to_day)
    # IFS setting: Times = 00:00 and 12:00, Steps = 12h # Is is a "forecast".
    # Variables:
    # 2 metre dewpoint temperature
    # 2 metre temperature
    # 10 metre U wind component
    # 10 metre V wind component
    # 10 metre wind gust since previous post-processing
    # Maximum temperature at 2 metres since previous post-processing
    # Minimum temperature at 2 metres since previous post-processing
    # Mean sea level pressure
    # Sea surface temperature
    # Snow depth
    # Snowfall
    # Sunshine duration
    # Top net solar radiation
    # Total precipitation
    command_multi = ({
        "class": "ei",
        "dataset": "interim",
        "date": t_interval,
        "expver": "1",
        "grid": "0.75/0.75",
        "levtype": "sfc",
        "param": "34.128/49.128/141.128/144.128/151.128/165.128/166.128/167.128/168.128/178.128/189.128/201.128/202.128/228.128",
        "step": "12",
        "stream": "oper",
        "time": "00:00:00/12:00:00",
        "format": "netcdf",
        "type": "fc",
        "target": "%s/%s" % (downloadDir, filename_multivar),
    })
    server.retrieve(req=command_multi)


def getShcws(from_day, to_day, downloadDir, server):
    # Significant height of combined wind waves and swell
    filename_shcws = 'era-int_shcws_%s_to_%s.nc' % (from_day, to_day)
    print('PROCESSING %s...' % (filename_shcws))
    t_interval = "%s/to/%s" % (from_day, to_day)
    # IFS setting: Times = 00:00, 06:00, 12:00, 18:00, Steps = 0h # It is an "analysis", not a forecast.
    command_shcws = {
        "class": "ei",
        "dataset": "interim",
        "date": t_interval,
        "expver": "1",
        "grid": "0.75/0.75",
        "levtype": "sfc",
        "param": "229.140",
        "step": "0",
        "stream": "wave",
        "time": "00:00:00/06:00:00/12:00:00/18:00:00",
        "format": "netcdf",
        "type": "an",
        "target": "%s/%s" % (downloadDir, filename_shcws),
    }
    server.retrieve(req=command_shcws)


def getMultivarMon(from_day, to_day, downloadDir, server):
    # get MULTIPLE VAIABLES :
    filename_multivar1 = 'era-int_multivarm1_%s_to_%s.nc' % (from_day, to_day)
    filename_multivar2 = 'era-int_multivarm2_%s_to_%s.nc' % (from_day, to_day)
    print("PROCESSING %s..." % (filename_multivar1))
    
    yrs = list(range(from_day.year, to_day.year+1))
    mts = list(['01-01', '02-01', '03-01', '04-01', '05-01', '06-01', '07-01',
                '08-01', '09-01', '10-01', '11-01', '12-01'])
    comb = pd.DataFrame(list(itertools.product(yrs, mts)))
    dts = pd.DataFrame({'yrs': comb[0], 'mts': comb[1]})
    dts['combi'] = dts.yrs.astype(str).str.cat(dts.mts.astype(str), sep='-')
    dts = dts.assign(date=dts.combi.apply(lambda x:
                                           datetime.strptime(x, '%Y-%m-%d')))
    dts = dts.assign(datestr=dts.date.apply(lambda x:
                                            '%s%s%s' %
                                            (x.year,
                                             '%02d' % x.month,
                                             '%02d' % x.day)))
    dts = dts.query('date<=@to_day')
    datestring = "/".join(dts.datestr)

    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": datestring,
        "expver": "1",
        "grid": "0.75/0.75",
        "levtype": "sfc",
        "param": "31.128/34.128/35.128/134.128/139.128/151.128/165.128/166.128/167.128/168.128/174.128/186.128/187.128/188.128/207.128/235.128",
        "stream": "moda",
        "type": "an",
        "format": "netcdf",
        "target": "%s/%s" % (downloadDir, filename_multivar1),
    })
    print("PROCESSING %s..." % (filename_multivar2))
    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": datestring,
        "expver": "1",
        "grid": "0.75/0.75",
        "levtype": "sfc",
        "param": "159.128/231.128/232.128",
        "stream": "moda",
        "format": "netcdf",
        "type": "fc",
        "target":  "%s/%s" % (downloadDir, filename_multivar2),
    })
    
