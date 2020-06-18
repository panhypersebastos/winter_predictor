# This script runs the download and the insertion feeds for ERA INTERIM
# Last modification: massond 2017-10-05

# To run this code in a BATCH mode, enter the following command in the shell:
# python /home/dmasson/dev/data_management/ERA-INTERIM/era_interim_feed.py

import os
os.system('python era_interim_download_monthly.py')
# os.system('era_interim_insert.py')
