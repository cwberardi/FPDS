# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:56:13 2016

@author: Chris
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x:'%f'%x)

HDF5path = r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Data\FPDS.h5'

serviceMap = {'/as': 'DLA', 
              '/departmentofthearmy': "Army", 
              '/departmentoftheairforce': 'Air Force',
              '/other': 'Other DoD',
              '/departmentofthenavy': 'Navy'}
              
rncMap = {'UNQ': 'Unique Source',
          'FOC': 'Follow-on Contract',
          'UR': 'Unsolicited Research Proposal',
          'PDR': 'Patent or Data Rights',
          'UT' : 'Utilities',
          'STD' : 'Standardization',
          'ONE': 'One Source',
          'URG': 'Urgency',
          'MES' : 'Mobilization, Essential R&D',
          'IA' : 'International Agreement',
          'OTH': 'Authorized by Statute',
          'RES' : 'Authorized resale',
          'NS' : 'National Security',
          'PI' : 'Public Interest',
          'SP2' : 'SAP Non-Competition',
          'BND' : 'Brand Name',
          '8AN' : 'Small Business'}

catCols = ['GFE-GFP', 'comp_type', 'contractActionType','contractBundling','contractFinancing','costOrPricingData', 'typ_set_aside', 'Service', 
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
        service = x.split('_')[0]
        tempDF = store[x]
        tempDF = tempDF[tempDF.obligatedAmount > dollarAmt]
        tempDF['Service'] = service
        finalDF = pd.concat([finalDF, tempDF], ignore_index = False)
    store.close()
    return finalDF
#%%
#==============================================================================
# Number of values in clipDF should be 3,254,734
#==============================================================================
clipDF =  storeclip(HDF5path, 25000)
clipDF.Service = clipDF.Service.map(serviceMap)
makeCat(catCols, clipDF)
clipDF.set_index('signedDate', inplace=True)
clipDF.index = clipDF.index + pd.DateOffset(months=3)
clipDF['FY'] = clipDF.index.year
#%%
#==============================================================================
# TODO: Need to think through whether to resample given FY or use a dateoffset in the datetime index
# sample offset: clipDF.index.min() - pd.DateOffset(months=9)
#==============================================================================

clipDF.FY.value_counts().sort_index().plot(kind='bar')

#%%
testDF = pd.DataFrame()
testDF['awds'] = clipDF[clipDF.modNum == '0']['vendorName'].resample('M').count()
testDF['pdr'] = clipDF[(clipDF.modNum == '0') & (clipDF.reason_not_competed == 'PDR')].reason_not_competed.resample('M').count()
testDF['jnatot'] = clipDF[clipDF.modNum == '0'].reason_not_competed.resample('M').count()
testDF['ratiopdrtoawd'] = testDF.pdr.div(testDF.awds)
testDF['ratiojnatoawd'] = testDF.jnatot.div(testDF.awds)
testDF['ratiopdrtojna'] = testDF.pdr.div(testDF.jnatot)
#%%
#==============================================================================
# Trends for competition type over time
#==============================================================================
compDF = pd.crosstab(clipDF.FY, clipDF.comp_type)
compDF.index = pd.to_datetime(compDF.index, format = '%Y')
compDF.drop(['CDO', 'NDO'], axis = 1, inplace=True)
compDF = compDF.reindex(columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
compDF.plot(kind='area', stacked=True)
plt.legend(frameon=True,  bbox_to_anchor=(0., 1.01, 1., .102), loc=4,
           ncol=4, mode="expand", borderaxespad=0.)
#%%
#==============================================================================
# Ratio plots of PDR, JnA, and Awards
#==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7,8))
fig.suptitle('Ratios of Data Rights (PDR) to JnA and Awds')

testDF.ratiojnatoawd.plot(ax=ax1, color='g', label = 'JnA to total Awds')
testDF.ratiojnatoawd.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax1, style='r--')
testDF.ratiopdrtoawd.plot(ax=ax2, label = 'PDR to total Awds')
testDF.ratiopdrtoawd.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax2, style='r--')
testDF.ratiopdrtojna.plot(ax=ax3, label = 'PDR to total JnA')
testDF.ratiopdrtojna.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax3, style='r--')

ax2.legend(frameon=True, shadow=True)
ax1.legend(frameon=True, shadow=True)
ax3.legend(frameon=True, shadow=True)

#%%
#==============================================================================
# Number of reasons_not_competed by fiscal year
#==============================================================================
notcompDF = pd.crosstab(clipDF.FY, clipDF.reason_not_competed)
notcompDF.index = pd.to_datetime(notcompDF.index, format = '%Y')
notcompDF.sort_index(ascending=True, inplace=True)
notcompDF.plot(kind='area', stacked=True)
plt.legend(frameon=True,  bbox_to_anchor=(1.01, 0, .5, 0.5), loc=4,
           ncol=1, mode="expand", borderaxespad=0.)
#%%
#==============================================================================
# quantity of dollars obligated by reasons_not_competed and fiscal year
#==============================================================================
dollarDF = pd.crosstab(clipDF.FY, clipDF.reason_not_competed, clipDF.obligatedAmount, aggfunc = 'sum')
dollarDF.index = pd.to_datetime(dollarDF.index, format = '%Y')
dollarDF.rename(columns = rncMap, inplace=True)
dollarDF.plot(kind='area', stacked=True)
plt.legend(frameon=True,  bbox_to_anchor=(1.01, 0, .4, .102), loc=4,
           ncol=1, mode="expand", borderaxespad=0.)
           
#%%
#==============================================================================
# Top 15 PDR JnA Vendors and cumulative dollar amounts from FY08-FY15
#==============================================================================
clipDF.groupby('reason_not_competed').get_group('PDR').groupby('vendorName')['obligatedAmount'].sum().sort_values(ascending=False).head(n=15)