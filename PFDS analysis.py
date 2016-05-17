# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:56:13 2016

@author: Chris
"""
import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x:'%f'%x)

HDF5path = r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Data\FPDS.h5'

catCols = ['GFE-GFP', 'comp_type', 'contractActionType','contractBundling','contractFinancing','costOrPricingData', 'typ_set_aside', 
           'typeOfContractPricing','undefinitizedAction','vendorCOSmallBusDeter','statuteExcpToFairOp', 'solprocedures', 'reason_not_competed', 
           'agencyID', 'claimantProgramCode', 'fedBizOpps','foreignFunding','fundingReqAgencyID','fundingReqOfficeID', 'itCommercialItemCat', 'multiYearContract']

#%%
def makeCat(columns, df):
    for x in columns:
        df[x] = df[x].astype('category')
        
def storeclip(HDF5path, dollarAmt):
    
    assert type(dollarAmt)  == int, 'Must be an integer arguement'
    assert type(HDF5path)   == str, "Must be a path to HDF5 store"
    
    finalDF = pd.DataFrame()
    store = pd.HDFStore(HDF5path, mode='r')
    for x in store.keys():
        print(x)
        tempDF = store[x]
        finalDF = pd.concat([finalDF, tempDF[tempDF.obligatedAmount > dollarAmt]], ignore_index = False)
    store.close()
    return finalDF
#%%
#==============================================================================
# Need to try a higher value amount, like $25K
#==============================================================================
clipDF =  storeclip(HDF5path, 10000)
makeCat(catCols, clipDF)
clipDF.set_index('signedDate', inplace=True)
#%%

data = clipDF[clipDF.reason_not_competed == 'PDR'].reason_not_competed.resample('M').count()
data1 = clipDF.reason_not_competed.resample('M').count()

data.div(data1).plot()

