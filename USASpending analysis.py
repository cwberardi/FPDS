# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 08:22:59 2016

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

USASpendingpath = 'C:\\Users\\Chris\\Documents\\MIT\\Dissertation\\FPDS\\Data\\USASpending.h5'

crossVal = pd.read_csv(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Data\crossvalidationFY.csv', index_col=0).rename(columns= {'ObligatedAmount': 'dollarsobligated'})

#rnccats = list(pd.read_csv(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Git\Lookup Tables\ReasonNotCompleted.csv', skiprows=18)['NULL'].dropna().str.strip())
   
forfundmap = {'A': 'Foreign Funds FMS','B': 'Foreign Funds non-FMS','X': 'Not Applicable'}
              
dollarFormatter         = FuncFormatter(lambda x, pos:'${:,.0f}'.format(x))
millionFormatter        = FuncFormatter(lambda x, pos:'$%1.1fM' % (x*1e-6))
millionNDFormatter        = FuncFormatter(lambda x, pos:'%1.1fM' % (x*1e-6))
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
        tempDF = store[x]
        makeCat(['maj_agency_cat'], tempDF)      
        finalDF = pd.concat([finalDF, tempDF[tempDF.dollarsobligated != 0.0]], ignore_index = False)
    store.close()
    return finalDF
    
#%%
#==============================================================================
# Number of values in clipDF should be 11362361 with zero dollar obs removed
# TODO: Need to think through whether to resample given fiscal_year or use a dateoffset in the datetime index
# sample offset: clipDF.index.min() + pd.DateOffset(months=3)
#==============================================================================
clipDF = storeclip(USASpendingpath)
clipDF.signeddate = pd.to_datetime(clipDF.signeddate, infer_datetime_format=True)
clipDF.set_index('signeddate', inplace=True)
clipDF.index = clipDF.index + pd.DateOffset(months=3)
clipDF.replace(':', np.NaN, inplace=True)
clipDF.reasonnotcompeted = clipDF.reasonnotcompeted.str.strip()
clipDF['p_or_s']= clipDF.psc_cat.str.isdigit().map({True: 'Product', False :'Service'})
clipDF['Service']='OTHER DOD'
clipDF.loc[clipDF.contractingofficeagencyid == '2100: DEPT OF THE ARMY', 'Service']='DEPT OF THE ARMY'
clipDF.loc[clipDF.contractingofficeagencyid == '97AS: DEFENSE LOGISTICS AGENCY', 'Service']='DEFENSE LOGISTICS AGENCY'
clipDF.loc[clipDF.contractingofficeagencyid == '1700: DEPT OF THE NAVY', 'Service']='DEPT OF THE NAVY'
clipDF.loc[clipDF.contractingofficeagencyid == '5700: DEPT OF THE AIR FORCE', 'Service']='DEPT OF THE AIR FORCE'
makeCat(['contractingofficeagencyid','p_or_s','contractingofficeid','psc_cat', 'productorservicecode', 'principalnaicscode','fundedbyforeignentity',
         'extentcompeted', 'vendorname','reasonnotcompeted','Service'], clipDF)
clipDF.fundedbyforeignentity = clipDF.fundedbyforeignentity.map(forfundmap)
assert len(clipDF)==11362361, "Does not match validation number"
#%%
#==============================================================================
# Removing DLA 2015 outlier obligations.  If intent is to maintain database integrity skip this step, otherwise it is 
# necessary for comparison of JnA frequencies
#==============================================================================
clipDF.loc[(clipDF.contractingofficeagencyid=='97AS: DEFENSE LOGISTICS AGENCY')&(clipDF.fiscal_year==2015)&((clipDF.dollarsobligated<3000)&(clipDF.dollarsobligated>-3000)), 'dollarsobligated']=np.nan
clipDF.dropna(axis=0,subset=['dollarsobligated'], inplace=True)
assert len(clipDF)==9555267, "Does not match validation number"
#%%
#==============================================================================
# Creates percent error for each fiscal year against crossvalidation numbers from USASpending.gov.  
# Asserts error if outside 0.25% difference in any single fiscal year
#==============================================================================
millionFormatter        = FuncFormatter(lambda x, pos:'$%1.0fM' % (x*1e-6))
sns.set_style('ticks')
fig = plt.figure(figsize=(5,2))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
errorDF = pd.DataFrame()
errorDF['% Diff'] = (crossVal.dollarsobligated-clipDF.groupby('fiscal_year')['dollarsobligated'].sum()).div(crossVal.dollarsobligated)
errorDF['Difference'] = (crossVal.dollarsobligated-clipDF.groupby('fiscal_year')['dollarsobligated'].sum())
errorDF['% Diff'].plot(kind='bar', position=1, ax=ax, color = 'r', legend=True, label = '% Error',ylim=(0,0.0003), **{'width':0.3})
errorDF.Difference.plot(kind='bar', position=0, ax=ax2,ylim=(0,60000000), legend=True, label = 'Absolute Error [secondary y-axis]', **{'width':0.3})
ax2.legend(loc= 'upper left')
ax.legend(bbox_to_anchor= (0.286,0.85))
ax.yaxis.set_major_formatter(percentFormatter)
ax2.yaxis.set_major_formatter(millionFormatter)
ax.yaxis.set_ticks([0,0.0001,0.0002,0.0003])
ax2.yaxis.set_ticks([0,20000000,40000000,60000000])
ax2.tick_params(labelsize=9)
ax.tick_params(labelsize=9)
ax.set_title('USASpending.gov Crossvalidation Results', fontsize=12, y=1.04)
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\dataerrors.jpg', bbox_inches='tight',dpi=150)

assert all(abs((crossVal.dollarsobligated-clipDF.groupby('fiscal_year')['dollarsobligated'].sum()).div(crossVal.dollarsobligated)) < 0.0025), "Outside acceptable error threshold"
millionFormatter        = FuncFormatter(lambda x, pos:'$%1.1fM' % (x*1e-6))
#%%
#==============================================================================
# Heatmap of Other DoD agencies obligations
#==============================================================================

agID = pd.crosstab(clipDF.contractingofficeagencyid, clipDF.fiscal_year, clipDF.dollarsobligated, aggfunc='sum')
agID.fillna(0, inplace=True)
agID[agID<0]=-1
agID[agID==0]=0
agID[agID>0]=1
ax = sns.heatmap(agID, square=True, cmap= 'RdYlGn',linecolor='black', linewidth=0.05, **{'alpha':0.8})
cbar = ax.collections[0].colorbar
cbar.set_ticks([agID.min().min(), 0, agID.max().max()])
cbar.set_ticklabels(["Negative Obligations", "Zero Obligations", "Positive Obligations"])
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Obligations by Agency and Fiscal Year', fontsize=12,  y=1.05)
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\DoD Agency Scaled Obligations.jpg', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Histogram of obligation amounts
#==============================================================================
sns.set_style('ticks')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
fig.subplots_adjust(wspace=0.32) 
sns.distplot(clipDF.dollarsobligated, bins=160, kde=False, ax=ax1, hist_kws={'range':(-10e9, 10e9)})
ax1.set_yscale('log')
ax1.set_xticklabels([-2*10**9, -1*10**9, 0, 1*10**9, 2*10**9, 3*10**9, 4*10**9],rotation =45, horizontalalignment='right')
ax1.grid(alpha = 0.5)
ax1.xaxis.set_major_formatter(billionFormatter)
ax1.set_xlim((-2000000000,4000000000))
ax1.set_ylabel('Frequency (log)')
ax1.set_title('Histogram of Obligated Amounts\n($-10B > x > 10B$)', fontsize=14, y=1.03)

totobs = clipDF[clipDF.dollarsobligated>0].dollarsobligated.sum()
totdeobs = clipDF[clipDF.dollarsobligated<0].dollarsobligated.sum()
ax2.bar([-0.44,0], [totdeobs, totobs], tick_label=['Deobligations', 'Obligations'], align='center', color=['r','g'], width = 0.35, alpha=0.5)
ax2.yaxis.set_major_formatter(trillionFormatter)
ax2.axhline(color='0.2', lw=1.0)
ax2.set_xlim((-0.7,0.26))
ax2.set_ylabel('Amount Obligated')
ax2.set_title('Total Deobligations and Obligations\n(FY08-FY15)', fontsize=14, y=1.03)

fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\histbarobs.jpg', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# subplots for obligations by servie
#==============================================================================
sns.set_style(style='ticks',rc={'xtick.direction': u'in'})
billionFormatter        = FuncFormatter(lambda x, pos:'$%1.0fB' % (x*1e-9))
fig, axs = plt.subplots(2, 5, figsize=(8,3), sharex=True, sharey='row')
titlemap = {'DEPT OF THE AIR FORCE': 'Air Force', 'DEPT OF THE ARMY': 'Army', 'DEPT OF THE NAVY': 'Navy', 'OTHER DOD': 'Other DoD', 'DEFENSE LOGISTICS AGENCY': 'DLA'}

sub = pd.crosstab(clipDF.Service, clipDF.fiscal_year).T.reindex(columns=['DEPT OF THE AIR FORCE',
'DEPT OF THE ARMY', 'DEPT OF THE NAVY','DEFENSE LOGISTICS AGENCY', 'OTHER DOD']).plot(kind='bar', subplots=True, ax=axs[0], legend=False)
sub2=pd.crosstab(clipDF.Service, clipDF.fiscal_year, clipDF.dollarsobligated, aggfunc='sum').T.reindex(columns=['DEPT OF THE AIR FORCE',
'DEPT OF THE ARMY', 'DEPT OF THE NAVY','DEFENSE LOGISTICS AGENCY', 'OTHER DOD']).plot(kind='bar', yticks = [0,4e10,8e10,12e10,16e10],subplots=True, ax=axs[1], legend=False)
for x in axs.flatten(): x.set_title(titlemap[x.get_title()])
axs[0][0].yaxis.set_major_formatter(millionNDFormatter)
axs[0][0].set_ylabel('Contract Actions')
axs[1][0].set_ylabel('Total Obligated')
axs[1][0].yaxis.set_major_formatter(billionFormatter)
for x in axs[1]: x.set_xlabel(''),x.set_title('')
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\serviceobligatedAmtSubPlots.jpg', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Resampling data frame monthly and counting frequencies of PDR JnA, JnA, and Awds. Note, only using inital awards (mod # of 0).
#==============================================================================
testDF = pd.DataFrame()
testDF['awds'] = clipDF[clipDF.modnumber=='0'].fiscal_year.resample('M').count()
testDF['pdr'] = clipDF[(clipDF.modnumber=='0')&(clipDF.reasonnotcompeted == 'PDR: PATENT/DATA RIGHTS')].reasonnotcompeted.resample('M').count()
testDF['jnatot'] = clipDF[clipDF.modnumber=='0'].reasonnotcompeted.resample('M').count()
testDF['ratiopdrtoawd'] = testDF.pdr.div(testDF.awds)
testDF['ratiojnatoawd'] = testDF.jnatot.div(testDF.awds)
testDF['ratiopdrtojna'] = testDF.pdr.div(testDF.jnatot)
#%%
df = pd.DataFrame()
df['awds'] = clipDF.dollarsobligated.resample('M').sum()
df['pdr'] = clipDF[clipDF.reasonnotcompeted == 'PDR: PATENT/DATA RIGHTS'].dollarsobligated.resample('M').sum()
df['jnatot'] = clipDF.dropna(subset=['reasonnotcompeted']).dollarsobligated.resample('M').sum()
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
clipDF.fiscal_year.value_counts().sort_index().plot(kind='bar', ax=ax1, label= 'Contract Actions',rot=0, **{'width':0.8})
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

clipDF.groupby('fiscal_year')['dollarsobligated'].sum().plot(kind='bar', color='g', label= 'Obligated $', rot=0, ax=ax3, **{'width':0.8})
ax3.legend(frameon=True, shadow=True, loc='lower right')
ax3.set_xlabel('')
ax3.yaxis.set_major_formatter(billionFormatter)
ax3.set_ylabel('Obligated Amount')

clipDF.dollarsobligated.resample('M').sum().plot(ax=ax4, label = 'Obligated $', color = 'g')
ax4.legend(frameon=True, shadow=True)
ax4.set_xlabel('Fiscal Year')
ax4.yaxis.set_major_formatter(billionFormatter)
ax4.set_ylabel('Obligated Amount')

fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\entriesperyear.tiff', dpi=150, bbox_inches='tight')

#%%
#==============================================================================
# Trends for competition type over time
#==============================================================================
compDF = pd.crosstab(clipDF.fiscal_year, clipDF.extentcompeted)
compDF.index = pd.to_datetime(compDF.index, format = '%Y')
compDF.drop(['CDO: COMPETITIVE DELIVERY ORDER', 'NDO: NON-COMPETITIVE DELIVERY ORDER'], axis = 1, inplace=True)
compDF = compDF.reindex(columns = ['A: FULL AND OPEN COMPETITION', 'B: NOT AVAILABLE FOR COMPETITION', 'C: NOT COMPETED', 'D: FULL AND OPEN COMPETITION AFTER EXCLUSION OF SOURCES',
                                   'E: FOLLOW ON TO COMPETED ACTION', 'F: COMPETED UNDER SAP ', 'G: NOT COMPETED UNDER SAP   '])
compDF.plot(kind='area', stacked=True)
plt.legend(frameon=True,  bbox_to_anchor=(0., 1.01, 1., .102), loc=4,ncol=2, mode="expand", borderaxespad=0.)
#%%           
#==============================================================================
# Competition type as a percentage of total contracts by fiscal_year
#==============================================================================
compDF2=compDF.copy()          
for rows in compDF2.index:
    compDF2.ix[rows]=compDF2.ix[rows].div(compDF2.ix[rows].sum())

compDF2.sort_index(ascending=True).plot(kind='area', stacked=True)
plt.vlines('4/2013', 0, 1, color = 'r', linestyle='dashed',label = 'Sequestration Start')
plt.legend(frameon=True,  bbox_to_anchor=(0., -.36, 1., .102), loc=4,
           ncol=2, mode="expand", borderaxespad=0.)
plt.ylabel('% of Total Contracts')
plt.ylim((0,1))
plt.title('Percentage of Total Contracts by Competition Type', fontsize=14, y=1.03)
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\compTypebyYear', dpi=150, bbox_inches='tight')
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
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\J&Aratio.tiff', dpi=150, bbox_inches='tight')
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

ax2.legend(frameon=True, shadow=True, loc = 'upper left')
ax1.legend(frameon=True, shadow=True)
ax3.legend(frameon=True, shadow=True, loc = 'upper left')
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\ratio$Plots.tiff', dpi=150, bbox_inches='tight')
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
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\ratioPlots.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Number of reasons_not_competed by fiscal year
#==============================================================================

notcompDF = pd.crosstab(clipDF.fiscal_year, clipDF.reasonnotcompeted)
notcompDF.index = pd.to_datetime(notcompDF.index, format = '%Y')
notcompDF.sort_index(ascending=True, inplace=True)
notcompDF.plot(kind='area', stacked=True)
plt.legend(frameon=True,  bbox_to_anchor=(1.01, 0, .38, 0.38), loc=4,
           ncol=1, mode="expand", borderaxespad=0.)
plt.title('Reason Not Competed by Fiscal Year', y=1.03,fontsize=14)
plt.tight_layout()

plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\rncbyfiscal_year.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# quantity of dollars obligated by reasons_not_competed and fiscal year
#==============================================================================
dollarDF = pd.crosstab(clipDF.fiscal_year, clipDF.reasonnotcompeted, clipDF.dollarsobligated, aggfunc = 'sum')
dollarDF.index = pd.to_datetime(dollarDF.index, format = '%Y')
dollarDF.plot(kind='area', stacked=False)
plt.legend(frameon=True,  bbox_to_anchor=(1.01, 0, .4, .102), loc=4,
           ncol=1, mode="expand", borderaxespad=0.)
           
#%%
#==============================================================================
# Pandas option for displaying floats as currencies
#==============================================================================
pd.options.display.float_format = '${:,.0f}'.format
np.set_printoptions(formatter={'float_kind':'${:,.0f}'})
#%%
#==============================================================================
# Top 15 PDR JnA Vendors and cumulative dollar amounts from fiscal_year08-fiscal_year15
#==============================================================================
clipDF.loc[clipDF.vendorname=='BOEING AEROSPACE OPERATIONS INCORPORATED', 'vendorname']='BOEING COMPANY, THE'
clipDF.loc[clipDF.vendorname=='BOEING AEROSPACE OPERATIONS, INC.', 'vendorname']='BOEING COMPANY, THE'               
clipDF.groupby('reasonnotcompeted').get_group('PDR: PATENT/DATA RIGHTS').groupby('vendorname')['dollarsobligated'].agg(['count', np.sum,np.mean]).sort_values(by='sum',ascending=False).head(n=15)

#%%
#==============================================================================
# Top 15 PDR JnA NAICS codes and cumulative dollar amount
#==============================================================================
clipDF.groupby('reasonnotcompeted').get_group('PDR: PATENT/DATA RIGHTS').groupby('principalnaicscode')['dollarsobligated'].agg(['count', np.sum, np.mean, np.std]).sort_values(by='sum', ascending=False).head(n=15)

#%%
#==============================================================================
# Reason not competed aggregation table
#==============================================================================
clipDF.groupby('reasonnotcompeted')['dollarsobligated'].agg(['count',np.sum, np.median, np.mean, np.std]).sort_values(by='count', ascending=False)

#%%
#==============================================================================
# Boxplot of Reason Not completed by Obligated Amount
#==============================================================================
plt.figure(figsize=(9,6))
sns.boxplot('reasonnotcompeted', 'dollarsobligated', 
            data=clipDF,whis=[5, 95], 
            order = clipDF.groupby('reasonnotcompeted')['dollarsobligated'].mean().sort_values( ascending=False).index.values,
            **{'showmeans': True})
plt.yscale('log')
plt.xticks(rotation=45, horizontalalignment = 'right')
plt.xlabel('')
plt.ylabel('Obligated Amount ($)')
plt.grid()
plt.tight_layout()
plt.title('Reason Not Competed by Obligated Amount ($)', y=1.03,fontsize=14)
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\rncby$.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Subplots, boxplot of PDR over fiscal_year and line plot of total PDR obligations and PDR frequency per fiscal_year
#==============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(9,8))

tableDF = pd.DataFrame()
tableDF['sum']= clipDF.groupby('reasonnotcompeted').get_group('PDR: PATENT/DATA RIGHTS').groupby('fiscal_year')['dollarsobligated'].sum()
tableDF['count']= clipDF[clipDF.modnumber=='0'].groupby('reasonnotcompeted').get_group('PDR: PATENT/DATA RIGHTS').groupby('fiscal_year')['dollarsobligated'].count()

sns.boxplot('fiscal_year', 'dollarsobligated', 
            data=clipDF[clipDF.reasonnotcompeted=='PDR: PATENT/DATA RIGHTS'],whis=[5, 95], 
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
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\PDR.tiff', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Subplots, PDR frequency by service and PDR total $ by service
#==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8,7))
fig.subplots_adjust(hspace=0.31)
fig.suptitle('PDR Frequency and Dollar Amount by Service',fontsize=14)
pd.crosstab(clipDF[clipDF.modnumber=='0'].Service, clipDF[clipDF.modnumber=='0'].reasonnotcompeted)['PDR: PATENT/DATA RIGHTS'].sort_values(ascending=True).plot(kind='barh', ax=ax1)
ax1.xaxis.set_major_formatter(generalFormatter)

pd.crosstab(clipDF.Service, clipDF.reasonnotcompeted, clipDF.dollarsobligated, aggfunc= np.sum)['PDR: PATENT/DATA RIGHTS'].sort_values().plot(kind='barh', color='g', ax=ax2)
ax2.xaxis.set_major_formatter(billionFormatter)

fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\PDRbyservice.tiff', dpi=150, bbox_inches='tight')

#%%
#==============================================================================
# Transforming FMS data using language from contract requirements description 
#==============================================================================
clipDF['FMS'] = clipDF.descriptionofcontractrequirement.str.contains('FMS')
clipDF.loc[clipDF.FMS==True, 'fundedbyforeignentity']= 'Foreign Funds FMS'

#%%
#==============================================================================
# Creates a choropleth of PDR obligations by state
#==============================================================================
testdata = pd.DataFrame(pd.crosstab(clipDF.K_office_state, clipDF.reasonnotcompeted)['PDR: PATENT/DATA RIGHTS'].drop(['#', 'AA', 'AE','AP','GU', 'VI', "ON", "PR",]))
testdata.index = testdata.index.astype(str)
state=pd.pivot_table(clipDF,index=['K_office_state','Office Name',], columns= 'reasonnotcompeted', values = 'dollarsobligated', aggfunc='count', dropna=True)['PDR: PATENT/DATA RIGHTS'].dropna().sort_values(ascending=False).astype(int)
for x in testdata.index:
    testdata.loc[x, 'text']=state[x].to_string(header=False, max_rows=5)

trc=[dict(
     type='choropleth',
     locations=testdata.index.values,
     locationmode='USA-states',
     colorscale=[[0,"rgb(220,220,220)"],[0.2,"rgb(245,195,157)"],[0.4,"rgb(245,160,105)"],[1,"rgb(178,10,28)"]],
     text=testdata.index.values + testdata.text,
     z=testdata['PDR: PATENT/DATA RIGHTS'].values.astype(float))]
     
lyt=dict(geo=dict(scope='usa', showlakes=True, showrivers=True), title="Patent and Data Rights Awards by State (fiscal_year08-fiscal_year15)")

pleth=go.Figure(data=trc,layout=lyt)
plot_url=py.plot(pleth, filename="Choropleth")