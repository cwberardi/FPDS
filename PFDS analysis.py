# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:56:13 2016

@author: Chris
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter
pd.set_option('display.float_format', lambda x:'%f'%x)

HDF5path = r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Data\FPDS.h5'

serviceMap = {'/as': 'DLA', 
              '/departmentofthearmy': "Army", 
              '/departmentoftheairforce': 'Air Force',
              '/other': 'Other DoD',
              '/departmentofthenavy': 'Navy'}
              
compMap= {'A': 'Full and Open Competition',
          'B': 'Not Available for Competition',
          'C': 'Not Competed',
          'D': 'Full and Open after Exclusion',
          'E': 'Follow On to Competed Action',
          'F': 'Competed under SAP',
          'G': 'Not Competed under SAP'}
              
rncMap = {'UNQ' : 'Unique Source',
          'FOC' : 'Follow-on Contract',
          'UR'  : 'Unsolicited Research Proposal',
          'PDR' : 'Patent or Data Rights',
          'UT'  : 'Utilities',
          'STD' : 'Standardization',
          'ONE' : 'One Source',
          'URG' : 'Urgency',
          'MES' : 'Mobilization, Essential R&D',
          'IA'  : 'International Agreement',
          'OTH' : 'Authorized by Statute',
          'RES' : 'Authorized resale',
          'NS'  : 'National Security',
          'PI'  : 'Public Interest',
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
# Number of values in clipDF should be 3,254,734 at a threshold of $25K
# TODO: Need to think through whether to resample given FY or use a dateoffset in the datetime index
# sample offset: clipDF.index.min() + pd.DateOffset(months=3)
#==============================================================================
clipDF =  storeclip(HDF5path, 25000)
clipDF.Service = clipDF.Service.map(serviceMap)
makeCat(catCols, clipDF)
clipDF.set_index('signedDate', inplace=True)
clipDF.index = clipDF.index + pd.DateOffset(months=3)
clipDF['FY'] = clipDF.index.year

#%%
#==============================================================================
# Resampling data frame monthly and counting frequencies of PDR JnA, JnA, and Awds. Note, only using inital awards (mod # of 0).
#==============================================================================
testDF = pd.DataFrame()
testDF['awds'] = clipDF['vendorName'].resample('M').count()
testDF['pdr'] = clipDF[(clipDF.reason_not_competed == 'PDR')].reason_not_competed.resample('M').count()
testDF['jnatot'] = clipDF.reason_not_competed.resample('M').count()
testDF['ratiopdrtoawd'] = testDF.pdr.div(testDF.awds)
testDF['ratiojnatoawd'] = testDF.jnatot.div(testDF.awds)
testDF['ratiopdrtojna'] = testDF.pdr.div(testDF.jnatot)
#%%
#==============================================================================
# Displays the frequency of entries by fiscal year in bar and line format with BBP initatives overlaid
#==============================================================================
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=False, figsize=(16,8))   
fig.suptitle('Number of Entries per Fiscal Year', fontsize=14, y=1.02)
clipDF.FY.value_counts().sort_index().plot(kind='bar', ax=ax1, label= 'Contract Actions', **{'width':0.8})
ax1.legend(frameon=True, shadow=True)

testDF.awds.plot(ax=ax2, label='Contract Actions')
ax2.annotate('BBP 3.0', xy=('4/2015', testDF.loc['4/2015', 'awds'][0]),  xycoords='data',
            xytext=('4/2015', testDF.loc['4/2015', 'awds'][0]+15000), textcoords='data',
            arrowprops=dict(facecolor='0.3', shrink=0.05, width=1, headwidth=9),
            horizontalalignment='center', verticalalignment='top',
             bbox=dict(boxstyle="round", fc="lightgrey"))
ax2.annotate('BBP 2.0', xy=('4/2013', testDF.loc['4/2013', 'awds'][0]),  xycoords='data',
            xytext=('4/2013', testDF.loc['4/2013', 'awds'][0]+15000), textcoords='data',
            arrowprops=dict(facecolor='0.3', shrink=0.05, width=1, headwidth=9),
            horizontalalignment='center', verticalalignment='top',
            bbox=dict(boxstyle="round", fc="lightgrey"))
ax2.annotate('BBP 1.0', xy=('6/2010', testDF.loc['6/2010', 'awds'][0]),  xycoords='data',
            xytext=('6/2010', testDF.loc['6/2010', 'awds'][0]+15000), textcoords='data',
            arrowprops=dict(facecolor='0.3', shrink=0.05, width=1, headwidth=9),
            horizontalalignment='center', verticalalignment='top',
            bbox=dict(boxstyle="round", fc="lightgrey"))
ax2.set_xlabel('Fiscal Year')
ax2.legend(frameon=True, shadow=True)

clipDF.groupby('FY')['obligatedAmount'].sum().plot(kind='bar', color='g', label= 'Obligated $', ax=ax3, **{'width':0.8})
ax3.legend(frameon=True, shadow=True)
ax3.set_xlabel('')
ax3.set_ylabel('Obligated Amount ($B)')
clipDF.obligatedAmount.resample('M').sum().plot(ax=ax4, label = 'Obligated $', color = 'g')
ax4.legend(frameon=True, shadow=True)
ax4.set_xlabel('Fiscal Year')
ax4.set_ylabel('Obligated Amount ($B)')

fig.tight_layout()
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\entriesperyear.tiff', dpi=150, bbox_inches='tight')

#%%
df = pd.DataFrame()
df['awds'] = clipDF.obligatedAmount.resample('M').sum()
df['pdr'] = clipDF[clipDF.reason_not_competed == 'PDR'].obligatedAmount.resample('M').sum()
df['jnatot'] = clipDF.dropna(subset=['reason_not_competed']).obligatedAmount.resample('M').sum()
df['ratiopdrtoawd'] = df.pdr.div(df.awds)
df['ratiojnatoawd'] = df.jnatot.div(df.awds)
df['ratiopdrtojna'] = df.pdr.div(df.jnatot)
#%%
#==============================================================================
# Trends for competition type over time
#==============================================================================
compDF = pd.crosstab(clipDF.FY, clipDF.comp_type)
compDF.index = pd.to_datetime(compDF.index, format = '%Y')
compDF.drop(['CDO', 'NDO'], axis = 1, inplace=True)
compDF = compDF.reindex(columns = ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
compDF.rename(columns = compMap, inplace=True)
compDF.plot(kind='area', stacked=True)
plt.legend(frameon=True,  bbox_to_anchor=(0., 1.01, 1., .102), loc=4,
           ncol=2, mode="expand", borderaxespad=0.)
#%%           
#==============================================================================
# Competition type as a percentage of total contracts by FY
#==============================================================================
compDF2=compDF.copy()          
for rows in compDF2.index:
    compDF2.ix[rows]=compDF2.ix[rows].div(compDF2.ix[rows].sum())

compDF2.rename(columns = compMap, inplace=True)
compDF2.sort_index(ascending=True).plot(kind='area', stacked=True)
plt.vlines('4/2013', 0, 1, color = 'r', linestyle='dashed',label = 'Sequestration Start')
plt.legend(frameon=True,  bbox_to_anchor=(0., -.36, 1., .102), loc=4,
           ncol=2, mode="expand", borderaxespad=0.)
plt.ylabel('% of Total Contracts')
plt.title('Percentage of Total Contracts by Competition Type', fontsize=14, y=1.03)
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\compTypebyYear', dpi=150, bbox_inches='tight')
#%%         
#==============================================================================
# Subplots, JnA ratio to Awds in both dollars and frequency & PDR ratio to JnA in both dollars and frequency  
#==============================================================================
sns.set_style('ticks')
fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(8,7))   
fig.suptitle('Ratio of JnA to Awards and PDR to JnA', y=1.03, fontsize=14)    
testDF.ratiojnatoawd.plot(color='g', label = 'JnA to total Awds', ax=ax1)
testDF.ratiojnatoawd.ewm(span=12).mean().plot(label='EWMA (12-month)',  ax=ax1, style='r--')
df.ratiojnatoawd.plot(color='b', label = 'JnA to total Awds ($)', ax=ax1)
df.ratiojnatoawd.ewm(span=12).mean().plot(label='EWMA (12-month)', ax=ax1, style='r-.')
ax1.set_ylim((0,0.65))
ax1.grid()
ax1.legend(frameon=True, shadow=True, ncol=2, loc='lower left')

testDF.ratiopdrtojna.plot(color='g', label = 'PDR to total JnA', ax=ax3)
testDF.ratiopdrtojna.ewm(span=12).mean().plot(label='EWMA (12-month)',  ax=ax3, style='r--')
df.ratiopdrtojna.plot(color='b', label = 'PDR to total JnA ($)', ax=ax3)
df.ratiopdrtojna.ewm(span=12).mean().plot(label='EWMA (12-month)', ax=ax3, style='r-.')
ax3.legend(frameon=True, shadow=True, ncol=2, loc='upper left')
ax3.set_xlabel('Fiscal Year')
ax3.grid()
plt.gcf().autofmt_xdate()

fig.tight_layout()
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\J&Aratio.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Ratio plots of PDR, JnA, and Awards by dollar amount
#==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7,8))
fig.suptitle('Ratios of Data Rights (PDR) to JnA and Awds')

df.ratiojnatoawd.plot(ax=ax1, color='g', label = 'JnA to total Awds')
df.ratiojnatoawd.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax1, style='r--')
df.ratiopdrtoawd.plot(ax=ax2, label = 'PDR to total Awds')
df.ratiopdrtoawd.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax2, style='r--')
df.ratiopdrtojna.plot(ax=ax3, label = 'PDR to total JnA')
df.ratiopdrtojna.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax3, style='r--')

ax2.legend(frameon=True, shadow=True)
ax1.legend(frameon=True, shadow=True)
ax3.legend(frameon=True, shadow=True)
#%%
#==============================================================================
# Ratio plots of PDR, JnA, and Awards by frequency
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

notcompDF = pd.crosstab(clipDF.FY[clipDF.modNum=="0"], clipDF.reason_not_competed[clipDF.modNum=='0'])
notcompDF.rename(columns=rncMap, inplace=True)
notcompDF.index = pd.to_datetime(notcompDF.index, format = '%Y')
notcompDF.sort_index(ascending=True, inplace=True)
notcompDF.plot(kind='area', stacked=True)
plt.legend(frameon=True,  bbox_to_anchor=(1.01, 0, .38, 0.38), loc=4,
           ncol=1, mode="expand", borderaxespad=0.)
plt.title('Reason Not Competed by Fiscal Year', y=1.03,fontsize=14)
plt.tight_layout()

plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\rncbyFY.tiff', dpi=150, bbox_inches='tight')
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
clipDF.groupby('reason_not_competed').get_group('Patent or Data Rights').groupby('vendorName')['obligatedAmount'].agg(['count', np.sum,np.mean]).sort_values(by='sum',ascending=False).head(n=15).to_clipboard()

#%%
#==============================================================================
# Top 15 PDR JnA NAICS codes and cumulative dollar amount
#==============================================================================
clipDF.groupby('reason_not_competed').get_group('Patent or Data Rights').groupby('principalNAICSCode')['obligatedAmount'].agg(['count', np.sum, np.mean, np.std]).sort_values(by='count', ascending=False).head(n=15)
#%%
#==============================================================================
# Pandas option for displaying floats as currencies
#==============================================================================
pd.options.display.float_format = '${:,.2f}'.format
np.set_printoptions(formatter={'float_kind':'${:,.2f}'})
#%%
#==============================================================================
# Reason not competed aggregation table
#==============================================================================
clipDF.groupby('reason_not_competed')['obligatedAmount'].agg(['count',np.sum, np.mean, np.std]).sort_values(by='count', ascending=False)

#%%
#==============================================================================
# Boxplot of Reason Not completed by Obligated Amount
#==============================================================================
plt.figure(figsize=(9,6))
sns.boxplot('reason_not_competed', 'obligatedAmount', 
            data=clipDF,whis=[5, 95], 
            order = clipDF.groupby('reason_not_competed')['obligatedAmount'].mean().sort_values( ascending=False).index.values,
            **{'showmeans': True})
plt.yscale('log')
plt.xticks(rotation=45, horizontalalignment = 'right')
plt.xlabel('')
plt.ylabel('Obligated Amount ($)')
plt.grid()
plt.tight_layout()
plt.title('Reason Not Competed by Obligated Amount ($)', y=1.03,fontsize=14)
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\rncby$.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Subplots, boxplot of PDR over FY and line plot of total PDR obligations and PDR frequency per FY
#==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(9,8))

yFormatter = FuncFormatter(lambda x, pos:'${:,.0f}'.format(x))
xFormatter = FuncFormatter(lambda x, pos:'%g'%x)
tableDF = clipDF.groupby('reason_not_competed').get_group('Patent or Data Rights').groupby('FY')['obligatedAmount'].agg(['count', np.sum])

sns.boxplot('FY', 'obligatedAmount', 
            data=clipDF[clipDF.reason_not_competed=='Patent or Data Rights'],whis=[5, 95], 
            ax=ax1,
            **{'showmeans': True})
ax1.set_yscale('log')
ax1.grid()
ax1.set_ylabel('Obligated Amount ($)')

tmpMean = tableDF['sum'].mean()
tableDF['sum'].plot(ax=ax2, label = 'Total Obligations',**{'marker':'.', 'ms':15})
ax2.axhline(y=tmpMean, color='r', linestyle='dashed',label = 'Average Obligation')
ax2.yaxis.set_major_formatter(yFormatter)
ax2.xaxis.set_major_formatter(xFormatter)
ax2.set_xlim((2007.5, 2015.5))
ax2.grid()
ax4 = ax2.twinx()
tableDF['count'].plot(ax=ax4, color = 'g', label = 'Frequency (secondary y)', **{'marker':'.', 'ms':15})
ax2.legend(shadow=True, frameon=True, loc='upper right')
ax4.legend(shadow=True, frameon=True, loc='lower left')
ax4.set_xlim((2007.5, 2015.5))

fig.autofmt_xdate()
fig.tight_layout()
fig.suptitle('Obligated Amounts for Patents and Data Rights by Fiscal Year', y=1.03,fontsize=14)
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\PDR.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Subplots, PDR frequency by service and PDR total $ by service
#==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8,7))
fig.suptitle('PDR Frequency and Dollar Amount by Service', y=1.03,fontsize=14)
pd.crosstab(clipDF.Service, clipDF.reason_not_competed)['Patent or Data Rights'].sort_values(ascending=True).plot(kind='barh', ax=ax1)

pd.crosstab(clipDF.Service, clipDF.reason_not_competed, clipDF.obligatedAmount, aggfunc= np.sum)['Patent or Data Rights'].sort_values().plot(kind='barh', color='g', ax=ax2)
ax2.xaxis.set_major_formatter(yFormatter)
fig.tight_layout()
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\PDRbyservice.tiff', dpi=150, bbox_inches='tight')
