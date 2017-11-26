from datetime import datetime, date
import numpy as np
import itertools
import pandas as pd

def getMultivarMon(from_day, to_day, downloadDir, server, resolution):
    # get MULTIPLE VARIABLES :
    filename_multivar1 = 'era-int_multivarm1_%s_%s_to_%s.nc' % (resolution, from_day, to_day)
    filename_multivar2 = 'era-int_multivarm2_%s_%s_to_%s.nc' % (resolution, from_day, to_day)
    filename_z = 'era-int_Zvar_%s_%s_to_%s.nc' % (resolution, from_day, to_day)
    
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

    print("PROCESSING %s..." % (filename_multivar1))
    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": datestring,
        "expver": "1",
        "grid": "%s/%s" % (resolution, resolution),
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
        "grid": "%s/%s" % (resolution, resolution),
        "levtype": "sfc",
        "param": "159.128/231.128/232.128",
        "stream": "moda",
        "format": "netcdf",
        "type": "fc",
        "target":  "%s/%s" % (downloadDir, filename_multivar2),
    })
    print("PROCESSING %s..." % (filename_z))
    server.retrieve({
        "class": "ei",
        "dataset": "interim",
        "date": datestring,
        "expver": "1",
        "grid": "%s/%s" % (resolution, resolution),
        "levelist": "70",
        "levtype": "pl",
        "param": "129.128",
        "stream": "moda",
        "type": "an",
        "format": "netcdf",
        "target":  "%s/%s" % (downloadDir, filename_z),
    })
    
