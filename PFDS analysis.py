# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:56:13 2016

@author: Chris
"""
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import FuncFormatter
import plotly.plotly as py
import plotly.graph_objs as go

py.sign_in('cwberardi', 'yk5snoxt1t')

pd.set_option('display.float_format', lambda x:'%f'%x)

HDF5path = r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Data\FPDS.h5'

agencyID = pd.Series(data = pd.read_csv(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Git\Lookup Tables\Agency_AgencyID.csv', 
                       usecols = [1,2], skip_blank_lines=True, skiprows = [1],index_col=0)['AgencyIDtext'], index =  pd.read_csv(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Git\Lookup Tables\Agency_AgencyID.csv', 
                       usecols = [1,2], skip_blank_lines=True, skiprows = [1],index_col=0).index).to_dict()

crossVal = pd.read_csv(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Data\crossvalidationFY.csv', index_col=0)

officeID = pd.read_csv(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Data\FPDSNG_Contracting_Offices.csv', index_col=4).to_dict('index')

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

dollarFormatter         = FuncFormatter(lambda x, pos:'${:,.0f}'.format(x))
millionFormatter        = FuncFormatter(lambda x, pos:'$%1.1fM' % (x*1e-6))
billionFormatter        = FuncFormatter(lambda x, pos:'$%1.1fB' % (x*1e-9))
trillionFormatter       = FuncFormatter(lambda x, pos:'$%1.1fT' % (x*1e-12))
generalFormatter        = FuncFormatter(lambda x, pos:'{0:,.0f}'.format(x))
percentFormatter        = FuncFormatter(lambda x, pos:'{:.2%}'.format(x))
threepercentFormatter   = FuncFormatter(lambda x, pos:'{:.3%}'.format(x))
#%%

def makePct(v, precision='0.2'):  
    """Convert number to percentage string."""
    return "{{:{}%}}".format(precision).format(v)

        
def makeCat(columns, df):
    for x in columns:
        df[x] = df[x].astype('category')
        
def makeDollar(x):
    return '${:,.0f}'.format(x)
    
def officeLookup(x, dictMap, ext):
    assert type(dictMap)==dict, 'Second variable must be a dictionary'
    try:
        j = dictMap[x][ext]
    except KeyError:
        j = np.nan
    return j
        
def storeclip(HDF5path):
    
    assert type(HDF5path)   == str, "Must be a path to HDF5 store"
    
    finalDF = pd.DataFrame()
    store = pd.HDFStore(HDF5path, mode='r')
    for x in store.keys():
        print(x)
        service = x.split('_')[0]
        tempDF = store[x].drop(['GFE-GFP', 'base/AllOptionsValue',
       'base/ExerOptValue', 'claimantProgramCode','contractActionType', 'contractBundling', 'contractFinancing',
       'costOrPricingData', 'currentCompletionDate', 'descripOfContReqS',
       'effectiveDate', 'fedBizOpps', 'forProfit', 'foreignFunding',
       'fundingReqAgencyID', 'fundingReqOfficeID', 'itCommercialItemCat',
       'modNumIDV', 'placeOfPerfState',
       'multiYearContract', 'numOfOffersReceived', 'perfBasedServiceK', 'piid', 
       'placeofPerfCongDist', 'placeofPerfZip','productOrServiceCode', 'solprocedures',
       'statuteExcpToFairOp', 'systemEquipmentCode', 'transNum', 'piidIDV', 
       'typ_set_aside', 'typeOfContractPricing', 'ultimateCompletionDate',
       'undefinitizedAction', 'vendorCOSmallBusDeter', 'vendorDUNS',
       'vendorLocCongDist', 'vendorLocState', 'vendorLocZip'], axis=1)
        tempDF['Service'] = service
        finalDF = pd.concat([finalDF, tempDF[tempDF.obligatedAmount != 0.0]], ignore_index = False)
    store.close()
    return finalDF
    
#%%
#==============================================================================
# Number of values in clipDF should be 12,695,755 and 11,004,363 with zero dollar obs removed
# TODO: Need to think through whether to resample given FY or use a dateoffset in the datetime index
# sample offset: clipDF.index.min() + pd.DateOffset(months=3)
#==============================================================================
clipDF = storeclip(HDF5path)
clipDF.Service = clipDF.Service.map(serviceMap)
clipDF.set_index('signedDate', inplace=True)
clipDF.index = clipDF.index + pd.DateOffset(months=3)
clipDF['FY'] = clipDF.index.year
clipDF['k_office_name'] = clipDF.k_OfficeID.map(lambda x: officeLookup(x,officeID,'CONTRACTING_OFFICE_NAME'))
clipDF['k_office_city'] = clipDF.k_OfficeID.map(lambda x: officeLookup(x,officeID,'ADDRESS_CITY'))
clipDF['k_office_state'] = clipDF.k_OfficeID.map(lambda x: officeLookup(x,officeID,'ADDRESS_STATE'))
clipDF['k_office_zip'] = clipDF.k_OfficeID.map(lambda x: officeLookup(x,officeID,'ZIP_CODE'))
makeCat(['Service', 'comp_type', 'reason_not_competed', 'k_OfficeAgencyID','k_OfficeID','agencyID','k_office_city','k_office_state', 'k_office_zip'], clipDF)
assert len(clipDF)==11004363, "Does not match validation number"
#%%
#==============================================================================
# Removing DLA 2015 outlier obligations.  If intent is to maintain database integrity skip this step, otherwise it is 
# necessary for comparison of JnA frequencies
#==============================================================================
clipDF.loc[(clipDF.Service=='DLA')&(clipDF.FY==2015)&((clipDF.obligatedAmount<2500)&(clipDF.obligatedAmount>-2500)), 'obligatedAmount']=np.nan
clipDF.dropna(axis=0,subset=['obligatedAmount'], inplace=True)
#%%
#==============================================================================
# Creates percent error for each fiscal year against crossvalidation numbers from USASpending.gov.  
# Asserts error if outside 0.25% difference in any single fiscal year
#==============================================================================
sns.set_style('ticks')
fig = plt.figure(figsize=(11,6))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
axins = zoomed_inset_axes(ax, 3.7, bbox_to_anchor = [760,410])
axins2= axins.twinx()
errorDF = pd.DataFrame()
errorDF['% Diff'] = (crossVal.ObligatedAmount-clipDF.groupby('FY')['obligatedAmount'].sum()).div(crossVal.ObligatedAmount)
errorDF['Difference'] = (crossVal.ObligatedAmount-clipDF.groupby('FY')['obligatedAmount'].sum())
errorDF['% Diff'].plot(kind='bar', position=1, ax=ax, color = 'r', ylim=(-0.00032,0.0026), **{'width':0.3})
errorDF.Difference.plot(kind='bar', position=0, ax=ax2, **{'width':0.3})
errorDF['% Diff'].plot(kind='bar', position=1, ax=axins, color = 'r', **{'width':0.3})
errorDF.Difference.plot(kind='bar', position=0, ax=axins2, **{'width':0.3})
axins.set_xlim(1.65,2.35)
axins2.set_xlim(1.65,2.35)
axins.set_ylim(0.00,-0.0001)
axins2.set_ylim((0,-6000000))
mark_inset(ax, axins, loc1=2, loc2=1, fc="0.5", alpha = 0.8, ec="black")
axins.set_yticks([0,-0.00005,-0.0001])
axins2.set_yticks([0,-2000000,-4000000,-6000000])
axins2.yaxis.set_major_formatter(millionFormatter)
ax.yaxis.set_major_formatter(percentFormatter)
axins.yaxis.set_major_formatter(threepercentFormatter)
ax2.yaxis.set_major_formatter(millionFormatter)
axins.tick_params(labelsize=7)
axins2.tick_params(labelsize=7)
axins.xaxis.set_visible(False)
axins.set_title('FY 2010', y=-.03)
ax.axhline(color='0.1',linewidth=1)
ax.set_title('Obligations Errors vs USASpending.gov data', fontsize=14, y=1.04)
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\dataerrors.tiff', bbox_inches='tight',dpi=150)

assert all(abs((crossVal.ObligatedAmount-clipDF.groupby('FY')['obligatedAmount'].sum()).div(crossVal.ObligatedAmount)) < 0.0045), "Outside acceptable error threshold"
#%%
#==============================================================================
# Heatmap of Other DoD agencies obligations
#==============================================================================

agID = pd.crosstab(clipDF.k_OfficeAgencyID, clipDF.FY, clipDF.obligatedAmount, aggfunc='sum')
agID.index = agID.index.to_series().map(agencyID)
agID.fillna(0, inplace=True)
agID[agID<0]=-1
agID[agID==0]=0
agID[agID>0]=1
ax = sns.heatmap(agID, square=True, cmap= 'RdYlGn',linecolor='black', linewidth=0.05, **{'alpha':0.8})
cbar = ax.collections[0].colorbar
cbar.set_ticks([agID.min().min(), 0, agID.max().max()])
cbar.set_ticklabels(["Negative Obligations", "Zero Obligations", "Positive Obligations"])
ax.set_title('Obligations by AgencyID and Fiscal Year', fontsize=14,  y=1.05)
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\DoD Agency Scaled Obligations.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Histogram of obligation amounts
#==============================================================================
sns.set_style('ticks')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))
fig.subplots_adjust(wspace=0.28) 
sns.distplot(clipDF.obligatedAmount, bins=160, kde=False, ax=ax1, hist_kws={'range':(-10e9, 10e9)})
ax1.set_yscale('log')
ax1.set_xticklabels([-2*10**9, -1*10**9, 0, 1*10**9, 2*10**9, 3*10**9, 4*10**9],rotation =45, horizontalalignment='right')
ax1.grid(alpha = 0.5)
ax1.xaxis.set_major_formatter(billionFormatter)
ax1.set_xlim((-2000000000,4000000000))
ax1.set_ylabel('Frequency (log)')
ax1.set_title('Histogram of Obligated Amounts ($-10B > x > 10B$)', fontsize=14, y=1.03)

totobs = clipDF[clipDF.obligatedAmount>0].obligatedAmount.sum()
totdeobs = clipDF[clipDF.obligatedAmount<0].obligatedAmount.sum()
ax2.bar([-0.44,0], [totdeobs, totobs], tick_label=['Deobligations', 'Obligations'], align='center', color=['r','g'], width = 0.35, alpha=0.5)
ax2.yaxis.set_major_formatter(trillionFormatter)
ax2.axhline(color='0.2', lw=1.0)
ax2.set_xlim((-0.7,0.26))
ax2.set_ylabel('Amount Obligated ($ Trillions)')
ax2.set_title('Total Deobligations and Obligations (FY08-FY15)', fontsize=14, y=1.03)

fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\histbarobs.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# subplots for obligatiosn by servie
#==============================================================================
sns.set_style(style='ticks',rc={'xtick.direction': u'in'})
pd.crosstab(clipDF.Service, clipDF.FY).T.plot(kind='bar', subplots=True, figsize=(8,8),legend=False, sharey=True,layout=(3,3))
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\serviceSubPlots.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# subplots for obligated amounts in dollars by service
#==============================================================================
pd.crosstab(clipDF.Service, clipDF.FY, clipDF.obligatedAmount, aggfunc='sum').T.plot(kind='bar', subplots=True, figsize=(8,8),legend=False, sharey=True,layout=(3,3))
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\serviceobligatedAmtSubPlots.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Resampling data frame monthly and counting frequencies of PDR JnA, JnA, and Awds. Note, only using inital awards (mod # of 0).
#==============================================================================
testDF = pd.DataFrame()
testDF['awds'] = clipDF[clipDF.modNum=='0'].vendorName.resample('M').count()
testDF['pdr'] = clipDF[(clipDF.modNum=='0')&(clipDF.reason_not_competed == 'PDR')].reason_not_competed.resample('M').count()
testDF['jnatot'] = clipDF[clipDF.modNum=='0'].reason_not_competed.resample('M').count()
testDF['ratiopdrtoawd'] = testDF.pdr.div(testDF.awds)
testDF['ratiojnatoawd'] = testDF.jnatot.div(testDF.awds)
testDF['ratiopdrtojna'] = testDF.pdr.div(testDF.jnatot)
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
# Displays the frequency of entries by fiscal year in bar and line format with BBP initatives overlaid
#==============================================================================
sns.set_style('ticks')
fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, sharex=False, figsize=(15,8))   
fig.suptitle('Number of Entries and Total Obligations per Fiscal Year', fontsize=14)
fig.subplots_adjust(wspace=0.26)
clipDF.FY.value_counts().sort_index().plot(kind='bar', ax=ax1, label= 'Contract Actions',rot=0, **{'width':0.8})
ax1.legend(frameon=True, shadow=True, loc='lower right')
ax1.yaxis.set_major_formatter(generalFormatter)

testDF.awds.plot(ax=ax2, label='Contract Actions')
ax2.annotate('BBP 3.0', xy=('4/2015', testDF.loc['4/2015', 'awds'][0]+9000),  xycoords='data',
            xytext=('4/2015', testDF.loc['4/2015', 'awds'][0]+80000), textcoords='data',
            arrowprops=dict(facecolor='0.3', shrink=0.05, width=1, headwidth=9),
            horizontalalignment='center', verticalalignment='top',
             bbox=dict(boxstyle="round", fc="lightgrey"))
ax2.annotate('BBP 2.0', xy=('4/2013', testDF.loc['4/2013', 'awds'][0]),  xycoords='data',
            xytext=('4/2013', testDF.loc['4/2013', 'awds'][0]+80000), textcoords='data',
            arrowprops=dict(facecolor='0.3', shrink=0.05, width=1, headwidth=9),
            horizontalalignment='center', verticalalignment='top',
            bbox=dict(boxstyle="round", fc="lightgrey"))
ax2.annotate('BBP 1.0', xy=('6/2010', testDF.loc['6/2010', 'awds'][0]),  xycoords='data',
            xytext=('6/2010', testDF.loc['6/2010', 'awds'][0]+80000), textcoords='data',
            arrowprops=dict(facecolor='0.3', shrink=0.05, width=1, headwidth=9),
            horizontalalignment='center', verticalalignment='top',
            bbox=dict(boxstyle="round", fc="lightgrey"))
ax2.set_xlabel('Fiscal Year')
ax2.yaxis.set_major_formatter(generalFormatter)
ax2.legend(frameon=True, shadow=True, loc='upper right')

clipDF.groupby('FY')['obligatedAmount'].sum().plot(kind='bar', color='g', label= 'Obligated $', rot=0, ax=ax3, **{'width':0.8})
ax3.legend(frameon=True, shadow=True, loc='lower right')
ax3.set_xlabel('')
ax3.yaxis.set_major_formatter(billionFormatter)
ax3.set_ylabel('Obligated Amount')

clipDF.obligatedAmount.resample('M').sum().plot(ax=ax4, label = 'Obligated $', color = 'g')
ax4.legend(frameon=True, shadow=True)
ax4.set_xlabel('Fiscal Year')
ax4.yaxis.set_major_formatter(billionFormatter)
ax4.set_ylabel('Obligated Amount')

fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\entriesperyear.tiff', dpi=150, bbox_inches='tight')


#%%         
#==============================================================================
# Subplots, JnA ratio to Awds in both dollars and frequency & PDR ratio to JnA in both dollars and frequency  
#==============================================================================
fig, (ax1, ax3) = plt.subplots(2, 1, sharex=True, figsize=(8,7))   
fig.suptitle('Ratio of JnA to Awards and PDR to JnA', y=1.03, fontsize=14)    
testDF.ratiojnatoawd.plot(color='g', label = 'JnA to total Awds', ax=ax1)
testDF.ratiojnatoawd.ewm(span=12).mean().plot(label='EWMA (12-month)',  ax=ax1, style='r--')
df.ratiojnatoawd.plot(color='b', label = 'JnA to total Awds ($)', ax=ax1)
df.ratiojnatoawd.ewm(span=12).mean().plot(label='EWMA (12-month)', ax=ax1, style='r-.')
ax1.grid()
ax1.legend(frameon=True, shadow=True, ncol=2, loc='upper right')

testDF.ratiopdrtojna.plot(color='g', label = 'PDR to total JnA', ax=ax3)
testDF.ratiopdrtojna.ewm(span=12).mean().plot(label='EWMA (12-month)',  ax=ax3, style='r--')
df.ratiopdrtojna.plot(color='b', label = 'PDR to total JnA ($)', ax=ax3)
df.ratiopdrtojna.ewm(span=12).mean().plot(label='EWMA (12-month)', ax=ax3, style='r-.')
ax3.legend(frameon=True, shadow=True, ncol=2, loc='upper left')
ax3.set_xlabel('Fiscal Year')
ax3.grid()

fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\J&Aratio.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Ratio plots of PDR, JnA, and Awards by dollar amount
#==============================================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7,8))
fig.suptitle('Ratios of Data Rights (PDR) to JnA and Awds')

df.ratiojnatoawd.plot(ax=ax1, color='g', label = 'JnA to total Awds ($)')
df.ratiojnatoawd.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax1, style='r--')
df.ratiopdrtoawd.plot(ax=ax2, label = 'PDR to total Awds ($)')
df.ratiopdrtoawd.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax2, style='r--')
df.ratiopdrtojna.plot(ax=ax3, label = 'PDR to total JnA ($)')
df.ratiopdrtojna.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax3, style='r--')

ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos:'{:.0%}'.format(x)))
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos:'{:.1%}'.format(x)))
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, pos:'{:.2%}'.format(x)))

ax2.legend(frameon=True, shadow=True)
ax1.legend(frameon=True, shadow=True)
ax3.legend(frameon=True, shadow=True)
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\ratio$Plots.tiff', dpi=150, bbox_inches='tight')
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

ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos:'{:.0%}'.format(x)))
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos:'{:.2%}'.format(x)))
ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, pos:'{:.2%}'.format(x)))

ax2.legend(frameon=True, shadow=True)
ax1.legend(frameon=True, shadow=True)
ax3.legend(frameon=True, shadow=True)
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\ratioPlots.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Number of reasons_not_competed by fiscal year
#==============================================================================

notcompDF = pd.crosstab(clipDF.FY, clipDF.reason_not_competed)
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
dollarDF.plot(kind='area', stacked=False)
plt.legend(frameon=True,  bbox_to_anchor=(1.01, 0, .4, .102), loc=4,
           ncol=1, mode="expand", borderaxespad=0.)
           
#%%
#==============================================================================
# Pandas option for displaying floats as currencies
#==============================================================================
pd.options.display.float_format = '${:,.0f}'.format
np.set_printoptions(formatter={'float_kind':'${:,.0f}'})
clipDF.reason_not_competed=clipDF.reason_not_competed.map(rncMap)
#%%
#==============================================================================
# Top 15 PDR JnA Vendors and cumulative dollar amounts from FY08-FY15
#==============================================================================
clipDF.loc[clipDF.vendorName=='BOEING AEROSPACE OPERATIONS INCORPORATED', 'vendorName']='BOEING COMPANY, THE'
clipDF.loc[clipDF.vendorName=='BOEING AEROSPACE OPERATIONS, INC.', 'vendorName']='BOEING COMPANY, THE'               
clipDF.groupby('reason_not_competed').get_group('PDR').groupby('vendorName')['obligatedAmount'].agg(['count', np.sum,np.mean]).sort_values(by='sum',ascending=False).head(n=15)

#%%
#==============================================================================
# Top 15 PDR JnA NAICS codes and cumulative dollar amount
#==============================================================================
clipDF.groupby('reason_not_competed').get_group('Patent or Data Rights').groupby('principalNAICSCode')['obligatedAmount'].agg(['count', np.sum, np.mean, np.std]).sort_values(by='sum', ascending=False).head(n=15)

#%%
#==============================================================================
# Reason not competed aggregation table
#==============================================================================
clipDF.groupby('reason_not_competed')['obligatedAmount'].agg(['count',np.sum, np.median, np.mean, np.std]).sort_values(by='count', ascending=False)

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

tableDF = pd.DataFrame()
tableDF['sum']= clipDF.groupby('reason_not_competed').get_group('Patent or Data Rights').groupby('FY')['obligatedAmount'].sum()
tableDF['count']= clipDF[clipDF.modNum=='0'].groupby('reason_not_competed').get_group('Patent or Data Rights').groupby('FY')['obligatedAmount'].count()

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
ax2.yaxis.set_major_formatter(billionFormatter)
ax2.grid()
ax4 = ax2.twinx()
tableDF['count'].plot(ax=ax4, color = 'g', label = 'Frequency (secondary y)', **{'marker':'.', 'ms':15})
ax2.legend(shadow=True, frameon=True, loc='lower left')
ax4.legend(shadow=True, frameon=True, loc='upper right')
ax4.xaxis.set_major_formatter(FuncFormatter(lambda x, pos:'%g'%x))
ax4.set_xlim((2007.5, 2015.5))
ax4.yaxis.set_major_formatter(generalFormatter)

fig.autofmt_xdate()
fig.tight_layout()
fig.suptitle('Obligated Amounts for Patents and Data Rights by Fiscal Year', y=1.03,fontsize=14)
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\PDR.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Subplots, PDR frequency by service and PDR total $ by service
#==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8,7))
fig.subplots_adjust(hspace=0.31)
fig.suptitle('PDR Frequency and Dollar Amount by Service',fontsize=14)
pd.crosstab(clipDF[clipDF.modNum=='0'].Service, clipDF[clipDF.modNum=='0'].reason_not_competed)['Patent or Data Rights'].sort_values(ascending=True).plot(kind='barh', ax=ax1)
ax1.xaxis.set_major_formatter(generalFormatter)

pd.crosstab(clipDF.Service, clipDF.reason_not_competed, clipDF.obligatedAmount, aggfunc= np.sum)['Patent or Data Rights'].sort_values().plot(kind='barh', color='g', ax=ax2)
ax2.xaxis.set_major_formatter(billionFormatter)

fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\PDRbyservice.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Creates a choropleth of PDR obligations by state
#==============================================================================
testdata = pd.DataFrame(pd.crosstab(clipDF.K_office_state, clipDF.reason_not_competed)['Patent or Data Rights'].drop(['#', 'AA', 'AE','AP','GU', 'VI', "ON", "PR",]))
testdata.index = testdata.index.astype(str)
state=pd.pivot_table(clipDF,index=['K_office_state','Office Name',], columns= 'reason_not_competed', values = 'obligatedAmount', aggfunc='count', dropna=True)['Patent or Data Rights'].dropna().sort_values(ascending=False).astype(int)
for x in testdata.index:
    testdata.loc[x, 'text']=state[x].to_string(header=False, max_rows=5)

trc=[dict(
     type='choropleth',
     locations=testdata.index.values,
     locationmode='USA-states',
     colorscale=[[0,"rgb(220,220,220)"],[0.2,"rgb(245,195,157)"],[0.4,"rgb(245,160,105)"],[1,"rgb(178,10,28)"]],
     text=testdata.index.values + testdata.text,
     z=testdata['Patent or Data Rights'].values.astype(float))]
     
lyt=dict(geo=dict(scope='usa', showlakes=True, showrivers=True), title="Patent and Data Rights Awards by State (FY08-FY15)")

pleth=go.Figure(data=trc,layout=lyt)
plot_url=py.plot(pleth, filename="Choropleth")