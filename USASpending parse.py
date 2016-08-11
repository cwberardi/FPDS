# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 12:46:07 2016

@author: Chris
"""
import pandas as pd
import numpy as np
 
cols = ['dollarsobligated',
 'maj_agency_cat',
 'contractingofficeagencyid',
 'contractingofficeid',
 'signeddate',
 'descriptionofcontractrequirement',
 'vendorname',
 'psc_cat',
 'productorservicecode',
 'principalnaicscode',
 'fundedbyforeignentity',
 'modnumber',
 'fiscal_year',
 'extentcompeted',
 'reasonnotcompeted',
 'numberofoffersreceived']
 
USASpendingpath = 'C:\\Users\\Chris\\Documents\\MIT\\Dissertation\\FPDS\\Data\\USASpending.h5'

dtypedict = {'dollarsobligated'                 : np.float64,
             'maj_agency_cat'                   : object, 
             'contractingofficeagencyid'        : object,
             'contractingofficeid'              : object,
             'descriptionofcontractrequirement' : str,
             'vendorname'                       : str,
             'fundedbyforeignentity'            : object,
             'psc_cat'                          : object,
             'productorservicecode'             : object,
             'principalnaicscode'               : object,
             'modnumber'                        : object,
             'fiscal_year'                      : int,
             'extentcompeted'                   : object,
             'reasonnotcompeted'                : object,
             'numberofoffersreceived'           : float}

filedict = {r'C:\Users\Chris\Downloads\2008_DOD_Contracts_Full_20160715.csv':'FY2008_20160715',
            r'C:\Users\Chris\Downloads\2009_DOD_Contracts_Full_20160715.csv':'FY2009_20160715',
            r'C:\Users\Chris\Downloads\2010_DOD_Contracts_Full_20160715.csv':'FY2010_20160715',
            r'C:\Users\Chris\Downloads\2011_DOD_Contracts_Full_20160715.csv':'FY2011_20160715',
            r'C:\Users\Chris\Downloads\2012_DOD_Contracts_Full_20160715.csv':'FY2012_20160715',
            r'C:\Users\Chris\Downloads\2013_DOD_Contracts_Full_20160715.csv':'FY2013_20160715',
            r'C:\Users\Chris\Downloads\2014_DOD_Contracts_Full_20160715.csv':'FY2014_20160715',
            r'C:\Users\Chris\Downloads\2015_DOD_Contracts_Full_20160715.csv':'FY2015_20160715'}

 #%%

def makeCat(columns, df):
    for x in columns:
        df[x] = df[x].astype('category')

def USASpendingHDF5(store_path, data_path, usecols, label, save=True):
    assert type(usecols)==list, "usecols must be a list" 
    
    tempDF = pd.read_csv(data_path, parse_dates=True, usecols = usecols, dtype = dtypedict, na_values=': ')
    
    if save == True:
        store = pd.HDFStore(store_path, complib='blosc', complevel=5)
        store[label]=tempDF  
        print(store)
        store.close()
    else:
        return tempDF
    
#%%
for x in filedict.items(): 
        USASpendingHDF5(USASpendingpath, x[0], cols, x[1])
