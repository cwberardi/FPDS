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
from matplotlib.ticker import FuncFormatter
import plotly.plotly as py
import plotly.graph_objs as go

sns.set_style('ticks')

plt.rc('font', family='serif')

params = {'xtick.labelsize': 12.0,
          'ytick.labelsize': 12.0,
          'legend.fontsize': 12.0,
          'text.usetex': True}
          
plt.rcParams.update(params)

py.sign_in('cwberardi', 'yk5snoxt1t')

pd.set_option('display.float_format', lambda x:'%f'%x)

USASpendingpath = 'C:\\Users\\Chris\\Documents\\MIT\\Dissertation\\FPDS\\Data\\USASpending.h5'

crossVal = pd.read_csv(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Data\crossvalidationFY.csv', index_col=0).rename(columns= {'ObligatedAmount': 'dollarsobligated'})

csisCats = pd.read_csv(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Git\Lookup Tables\ProductOrServiceCodes.csv', usecols = [0,4,6], index_col=0)

pscCats = pd.Series(data = csisCats.ProductOrServiceArea, index = csisCats.index)

simpleCats = pd.Series(data = csisCats.Simple, index = csisCats.index)

forfundmap = {'A': 'Foreign Funds FMS','B': 'Foreign Funds non-FMS','X': 'Not Applicable'}
              
dollarFormatter         = FuncFormatter(lambda x, pos:'\${:,.0f}'.format(x))
millionFormatter        = FuncFormatter(lambda x, pos:'\$%1.1fM' % (x*1e-6))
millionNDFormatter      = FuncFormatter(lambda x, pos:'%1.1fM' % (x*1e-6))
billionFormatter        = FuncFormatter(lambda x, pos:'\$%1.1fB' % (x*1e-9))
billionFormatterZero    = FuncFormatter(lambda x, pos:'\$%1.0fB' % (x*1e-9))
trillionFormatter       = FuncFormatter(lambda x, pos:'\$%1.1fT' % (x*1e-12))
generalFormatter        = FuncFormatter(lambda x, pos:'{0:,.0f}'.format(x))
percentFormatter        = FuncFormatter(lambda x, pos:'{:.2%}'.format(x))
threepercentFormatter   = FuncFormatter(lambda x, pos:'{:.3%}'.format(x))
latexpercentFormatter   = FuncFormatter(lambda x, pos: str(x*100)+'\%')
latexpercentFormatterZero = FuncFormatter(lambda x, pos: str(('{:,.0f}'.format(x*100)))+'\%')
#%%

def latex_float(f):
    float_str = "{0:.3g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0}e^{{{1}}}".format(base, int(exponent))
    else:
        return float_str

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
# Need to think through whether to resample given fiscal_year or use a dateoffset in the datetime index
# sample offset: clipDF.index.min() + pd.DateOffset(months=3)
#==============================================================================
clipDF = storeclip(USASpendingpath)
clipDF.signeddate = pd.to_datetime(clipDF.signeddate, infer_datetime_format=True)
clipDF.set_index('signeddate', inplace=True)
clipDF.index = clipDF.index + pd.DateOffset(months=3)
clipDF.replace(':', np.NaN, inplace=True)
clipDF.reasonnotcompeted = clipDF.reasonnotcompeted.str.strip()
clipDF['psc_only'] = clipDF.productorservicecode.str.split(':').str.get(0)
clipDF['psc_simple'] = clipDF.psc_only.map(simpleCats)
clipDF.psc_simple.replace('R&D', 'R\&D', inplace=True)
clipDF['psc_groups'] = clipDF.psc_only.map(pscCats)
clipDF['Service']='OTHER DOD'
clipDF.loc[clipDF.contractingofficeagencyid == '2100: DEPT OF THE ARMY', 'Service']='DEPT OF THE ARMY'
clipDF.loc[clipDF.contractingofficeagencyid == '97AS: DEFENSE LOGISTICS AGENCY', 'Service']='DEFENSE LOGISTICS AGENCY'
clipDF.loc[clipDF.contractingofficeagencyid == '1700: DEPT OF THE NAVY', 'Service']='DEPT OF THE NAVY'
clipDF.loc[clipDF.contractingofficeagencyid == '5700: DEPT OF THE AIR FORCE', 'Service']='DEPT OF THE AIR FORCE'
makeCat(['contractingofficeagencyid','contractingofficeid','psc_only','psc_simple', 'psc_simple','psc_cat', 
         'productorservicecode', 'principalnaicscode','fundedbyforeignentity',
         'extentcompeted', 'vendorname','reasonnotcompeted','Service'], clipDF)
clipDF.fundedbyforeignentity = clipDF.fundedbyforeignentity.map(forfundmap)
assert len(clipDF)==11362361, "Does not match validation number"
#%%
#==============================================================================
# Creates percent error for each fiscal year against crossvalidation numbers from USASpending.gov.  
# Asserts error if outside 0.25% difference in any single fiscal year
#==============================================================================
assert len(clipDF)==11362361, "Need to run on full set, not after DLA removal"

fig = plt.figure(figsize=(5,2))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
errorDF = pd.DataFrame()
errorDF['% Diff'] = (crossVal.dollarsobligated-clipDF.groupby('fiscal_year')['dollarsobligated'].sum()).div(crossVal.dollarsobligated)
errorDF['Difference'] = (crossVal.dollarsobligated-clipDF.groupby('fiscal_year')['dollarsobligated'].sum())
errorDF['% Diff'].plot(kind='bar', position=1, ax=ax, color = 'r', legend=True, label = '\% Error',ylim=(0,0.0005), **{'width':0.3})
errorDF.Difference.plot(kind='bar', position=0, ax=ax2,ylim=(0,80000000), legend=True, label = 'Absolute Error [secondary y-axis]', **{'width':0.3})
ax2.legend(loc= 'upper left')
ax.set_xlabel('')
ax.tick_params(top='off')
ax2.set_xlabel('')
ax.legend(bbox_to_anchor= (0.347,0.85))
ax.yaxis.set_major_formatter(latexpercentFormatter)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos:'\$%1.0fM' % (x*1e-6)))
ax.yaxis.set_ticks([0,0.0001,0.0002,0.0003, 0.0004, 0.0005])
ax2.yaxis.set_ticks([0,20000000,40000000,60000000, 80000000])
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\dataerrors.pdf', bbox_inches='tight')

assert all(abs((crossVal.dollarsobligated-clipDF.groupby('fiscal_year')['dollarsobligated'].sum()).div(crossVal.dollarsobligated)) < 0.0045), "Outside acceptable error threshold"

#%%
#==============================================================================
# subplots for obligations by servie
#==============================================================================
assert len(clipDF)==11362361, "Must not remove DLA obligations before proceeding"

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
axs[1][0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos:'\$%1.0fB' % (x*1e-9)))
for x in axs[1]: x.set_xlabel(''),x.set_title('')
for x in axs.flatten(): x.tick_params(axis='x', which=u'both',length=5, top='off')
for x in axs.flatten(): x.grid(alpha=0.6, axis='y')

fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\serviceobligatedAmtSubPlots.pdf', bbox_inches='tight')
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
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\DoD_Agency_Scaled_Obligations.pdf', dpi=600, bbox_inches='tight')
#%%
#==============================================================================
# Histogram of obligation amounts
#==============================================================================
assert len(clipDF)==9555267, "Need to remove DLA errors before running"

sns.distplot(clipDF.dollarsobligated, bins=160, kde=False, hist_kws={'range':(-10e9, 10e9)})
plt.gcf().set_size_inches(4.5, 4)
ax = plt.gca()
ax.set_yscale('log')
ax.set_xticklabels([-2*10**9, -1*10**9, 0, 1*10**9, 2*10**9, 3*10**9, 4*10**9],rotation =45, horizontalalignment='right')
ax.grid(alpha = 0.5)
plt.xlabel('')
ax.xaxis.set_major_formatter(billionFormatter)
ax.set_xlim((-2000000000,4000000000))
ax.set_ylabel('Frequency ($\log$)', fontsize=18)

plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\hist.pdf', bbox_inches='tight')

#%%
assert len(clipDF)==9555267, "Need to remove DLA errors before running"

totobs = clipDF[clipDF.dollarsobligated>0].dollarsobligated.sum()
totdeobs = clipDF[clipDF.dollarsobligated<0].dollarsobligated.sum()
plt.bar([-0.44,0], [totdeobs, totobs], tick_label=['Deobligations', 'Obligations'], align='center', color=['r','g'], width = 0.33, alpha=0.5)
plt.gcf().set_size_inches(4.5, 4.1)
ax=plt.gca()
ax.yaxis.set_major_formatter(trillionFormatter)
ax.axhline(color='0.2', lw=1.0)
ax.set_xlim((-0.7,0.26))
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\obs.pdf', bbox_inches='tight')

#%%
#==============================================================================
# Resampling data frame monthly and counting frequencies of PDR JnA, JnA, and Awds. Note, only using inital awards (mod # of 0).
#==============================================================================
assert len(clipDF)==9555267, "Need to remove DLA errors before running"

testDF = pd.DataFrame()
testDF['awds'] = clipDF[clipDF.modnumber=='0'].fiscal_year.resample('M').count()
testDF['pdr'] = clipDF[(clipDF.modnumber=='0')&(clipDF.reasonnotcompeted == 'PDR: PATENT/DATA RIGHTS')].reasonnotcompeted.resample('M').count()
testDF['jnatot'] = clipDF[clipDF.modnumber=='0'].reasonnotcompeted.resample('M').count()
testDF['ratiopdrtoawd'] = testDF.pdr.div(testDF.awds)
testDF['ratiojnatoawd'] = testDF.jnatot.div(testDF.awds)
testDF['ratiopdrtojna'] = testDF.pdr.div(testDF.jnatot)
#%%
assert len(clipDF)==9555267, "Need to remove DLA errors before running"

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

clipDF.groupby('fiscal_year')['dollarsobligated'].sum().plot(kind='bar', color='g', label= 'Obligated \$', rot=0, ax=ax3, **{'width':0.8})
ax3.legend(frameon=True, shadow=True, loc='lower right')
ax3.set_xlabel('')
ax3.yaxis.set_major_formatter(billionFormatterZero)
ax3.set_ylabel('Obligated Amount')

clipDF.dollarsobligated.resample('M').sum().plot(ax=ax4, label = 'Obligated \$', color = 'g')
ax4.legend(frameon=True, shadow=True)
ax4.set_xlabel('Fiscal Year')
ax4.yaxis.set_major_formatter(billionFormatterZero)
ax4.set_ylabel('Obligated Amount')

fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\entriesperyear.pdf', bbox_inches='tight')

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
fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\J&Aratio.pdf', dpi=150, bbox_inches='tight')
#%%
#==============================================================================
# Ratio plots of PDR, JnA, and Awards by dollar amount
#==============================================================================
fig, ((ax1, ax2) , (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex=True, figsize=(10,6.5))
fig.subplots_adjust(wspace=0.25)
#fig.suptitle('Ratios of Obligated Amounts (left column) and Ratio of Contract Awards (right column)')

df.ratiojnatoawd.plot(ax=ax1, color='g', label = 'JnA to Awds (\$)')
df.ratiojnatoawd.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax1,style='r-', alpha = 0.55, lw=3.0)
df.ratiopdrtoawd.plot(ax=ax3, color='g',label = 'PDR to Awds (\$)')
df.ratiopdrtoawd.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax3, style='r-', alpha = 0.55, lw=3.0)
df.ratiopdrtojna.plot(ax=ax5, color='g',label = 'PDR to JnA (\$)')
df.ratiopdrtojna.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax5, style='r-', alpha = 0.55, lw=3.0)

ax1.yaxis.set_major_formatter(latexpercentFormatterZero)
ax3.yaxis.set_major_formatter(latexpercentFormatter)
ax5.yaxis.set_major_formatter(latexpercentFormatter)

ax3.legend(frameon=True, shadow=True, loc = 'upper left', fontsize='x-small')
ax1.legend(frameon=True, shadow=True, fontsize='x-small')
ax5.legend(frameon=True, shadow=True, loc = 'upper left', fontsize='x-small')

ax1.set_ylabel('JnA to Total Awds')
ax3.set_ylabel('PDR to Total Awds')
ax5.set_ylabel('PDR to Total JnA')
ax5.set_xlabel('')
#==============================================================================
# Ratio plots of PDR, JnA, and Awards by frequency
#==============================================================================
#fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(7,8))
#fig.suptitle('Ratios of Data Rights (PDR) to JnA and Awds')

testDF.ratiojnatoawd.plot(ax=ax2,  label = 'JnA to Awds')
testDF.ratiojnatoawd.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax2, style='r-', alpha = 0.55, lw=3.0)
testDF.ratiopdrtoawd.plot(ax=ax4, label = 'PDR to Awds')
testDF.ratiopdrtoawd.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax4, style='r-', alpha = 0.55, lw=3.0)
testDF.ratiopdrtojna.plot(ax=ax6, label = 'PDR to JnA')
testDF.ratiopdrtojna.ewm(span=6).mean().plot(label='EWMA (6-month)', ax=ax6, style='r-', alpha = 0.55, lw=3.0)

ax2.yaxis.set_major_formatter(latexpercentFormatterZero)
ax4.yaxis.set_major_formatter(latexpercentFormatter)
ax6.yaxis.set_major_formatter(latexpercentFormatter)

ax4.legend(frameon=True, shadow=True, fontsize='x-small')
ax2.legend(frameon=True, shadow=True, fontsize='x-small')
ax6.legend(frameon=True, shadow=True, fontsize='x-small')

ax6.set_xlabel('')

fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\ratioPlots.pdf', dpi=600, bbox_inches='tight')

axlims = []
for a in (ax1, ax2 , ax3, ax4, ax5, ax6): axlims.append(a.get_ylim())
#%%
#==============================================================================
# Sub plots of OLS regressions on ratio plots
#==============================================================================
regList = (df.ratiojnatoawd,testDF.ratiojnatoawd, df.ratiopdrtoawd,testDF.ratiopdrtoawd,df.ratiopdrtojna,testDF.ratiopdrtojna)
fig, axs = plt.subplots(3, 2, sharex=True, figsize=(10,6.5))
fig.subplots_adjust(wspace=0.25)

regx = sm.add_constant(np.arange(1,97))
for reg, ax, lims, color in zip(regList, axs.flatten(), axlims, ('g', 'b', 'g','b', 'g','b')):
    regr = sm.OLS(reg, regx).fit()
#    print(regr.summary().as_latex()) # Only uncomment if updated summary tables are required
    label = '$y = ' +latex_float(regr.params[1])+'x + '+latex_float(regr.params[0])+ '$\n$r^2 = %.3f$' %  (regr.rsquared)
    sns.regplot(np.arange(1,97), reg,  ax=ax, color=color, scatter_kws={"alpha": 0.35}, line_kws={'alpha':0.9})
    ax.text(0.97,0.76,label, transform=ax.transAxes, fontsize=10, horizontalalignment = 'right', bbox=dict(facecolor='w', alpha=0.8))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos:'{:.1%}'.format(x)))
    ax.set_ylim(lims)
    ax.set_xlim((0,96))
    ax.set_ylabel('')

axs[0][0].yaxis.set_major_formatter(latexpercentFormatterZero)
axs[0][1].yaxis.set_major_formatter(latexpercentFormatterZero)
axs[1][0].yaxis.set_major_formatter(latexpercentFormatter)
axs[1][1].yaxis.set_major_formatter(latexpercentFormatter)
axs[2][0].yaxis.set_major_formatter(latexpercentFormatter)
axs[2][1].yaxis.set_major_formatter(latexpercentFormatter)

axs[0][0].set_ylabel('JnA to Total Awds')
axs[1][0].set_ylabel('PDR to Total Awds')
axs[2][0].set_ylabel('PDR to Total JnA')

axs[2][0].set_xlabel('Time (months)')
axs[2][1].set_xlabel('Time (months)')

axs[2][0].set_xticks(np.arange(0,85,12))
axs[2][1].set_xticks(np.arange(0,85,12))

fig.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\OLSsubplots.pdf', bbox_inches='tight')

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

plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\rncbyfiscal_year.pdf', bbox_inches='tight')
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
clipDF.groupby('reasonnotcompeted').get_group('PDR: PATENT/DATA RIGHTS').groupby('principalnaicscode')['dollarsobligated'].agg(['count', np.sum, np.mean]).sort_values(by='sum', ascending=False).head(n=15)
#%%
#==============================================================================
# PDR JnA psc_groups codes and cumulative dollar amount
#==============================================================================
clipDF.groupby('reasonnotcompeted').get_group('PDR: PATENT/DATA RIGHTS').groupby('psc_groups')['dollarsobligated'].agg(['count', np.sum, np.mean]).sort_values(by='sum', ascending=False)
#%%
#==============================================================================
# Reason not competed aggregation table
#==============================================================================
clipDF.groupby('reasonnotcompeted')['dollarsobligated'].agg(['count',np.sum, np.median, np.mean, np.std]).sort_values(by='sum', ascending=False)

#%%
#==============================================================================
# Boxplot of Reason Not completed by Obligated Amount
#==============================================================================
plt.figure(figsize=(9,5))
sns.boxplot('reasonnotcompeted', 'dollarsobligated', 
            data=clipDF,whis=[5, 95],
            order = clipDF.groupby('reasonnotcompeted')['dollarsobligated'].sum().sort_values(ascending=False).index.values,
            **{'showmeans': True})
plt.yscale('log')
plt.xticks(rotation=45, horizontalalignment = 'right')
plt.xlabel('')
plt.ylabel('Obligated Amount ($)')
plt.grid()
plt.tight_layout()
plt.title('Reason Not Competed by Obligated Amount ($)', y=1.03,fontsize=14)
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\rncby$.pdf', bbox_inches='tight')
#%%
#==============================================================================
# Subplots, boxplot of PDR over fiscal_year 
#==============================================================================
plt.figure(figsize=(10,4))
tableDF = pd.DataFrame()
tableDF['sum']= clipDF.groupby('reasonnotcompeted').get_group('PDR: PATENT/DATA RIGHTS').groupby('fiscal_year')['dollarsobligated'].sum()
tableDF['count']= clipDF[clipDF.modnumber=='0'].groupby('reasonnotcompeted').get_group('PDR: PATENT/DATA RIGHTS').groupby('fiscal_year')['dollarsobligated'].count()

sns.boxplot('fiscal_year', 'dollarsobligated', 
            data=clipDF[clipDF.reasonnotcompeted=='PDR: PATENT/DATA RIGHTS'],whis=[5, 95], color='lightblue',
            **{'showmeans': True})
ax1 = plt.gca()
ax1.set_yscale('log')
ax1.grid()
ax1.set_ylabel('Obligated Amount (\$)')
ax1.set_xlabel('')
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\PDRboxplot.pdf', bbox_inches='tight')
#%%
#==============================================================================
# Line plot of pdr frequency and total obligation by fiscal year
#==============================================================================
tmpMean = tableDF['sum'].mean()
tableDF['sum'].plot(label = 'Total Obligations', xlim = (2008,2015),**{'marker':'.', 'ms':15})
ax2 = plt.gca()
ax2.axhline(y=tmpMean, color='r', linestyle='dashed',label = 'Average Obligation')
ax2.yaxis.set_major_formatter(billionFormatter)
ax2.grid()
ax4 = ax2.twinx()
tableDF['count'].plot(ax=ax4, figsize = (10,4),color = 'g', label = 'Frequency (secondary y)', **{'marker':'.', 'ms':15})
ax2.legend(shadow=True, frameon=True, loc='lower left')
ax4.legend(shadow=True, frameon=True, loc='upper right')
ax4.xaxis.set_major_formatter(FuncFormatter(lambda x, pos:'%g'%x))
ax2.set_xlabel('')
ax4.set_xlim((2007.5, 2015.5))
ax4.yaxis.set_major_formatter(generalFormatter)

fig.autofmt_xdate()
fig.tight_layout()
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\PDRlineplot.pdf', dpi=600, bbox_inches='tight')
#%%
#==============================================================================
# Subplots, PDR frequency by service and PDR total $ by service
#==============================================================================

pd.crosstab(clipDF[clipDF.modnumber=='0'].Service, clipDF[clipDF.modnumber=='0'].reasonnotcompeted)['PDR: PATENT/DATA RIGHTS'].sort_values(ascending=True).plot(kind='barh')
ax = plt.gca()
plt.gcf().set_size_inches(6, 2)
plt.ylabel('')
ax.xaxis.set_major_formatter(generalFormatter)
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\PDRbyservice.pdf', bbox_inches='tight')

pd.crosstab(clipDF.Service, clipDF.reasonnotcompeted, clipDF.dollarsobligated, aggfunc= np.sum)['PDR: PATENT/DATA RIGHTS'].sort_values().plot(kind='barh', color='g')
plt.gcf().set_size_inches(6, 2)
ax2 = plt.gca()
plt.ylabel('')
ax2.xaxis.set_major_formatter(billionFormatter)

plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\PDRbyservicedollars.pdf', bbox_inches='tight')

#%%
#==============================================================================
# Transforming FMS data using language from contract requirements description 
#==============================================================================
clipDF['FMS'] = clipDF.descriptionofcontractrequirement.str.contains('FMS')
clipDF.loc[clipDF.FMS==True, 'fundedbyforeignentity']= 'Foreign Funds FMS'
#%%
params = {'xtick.labelsize': 14.0,
          'ytick.labelsize': 14.0,
          'legend.fontsize': 14.0}
plt.rcParams.update(params)
#==============================================================================
# Count of p_or_s by fiscal year and resulting line plot
#==============================================================================
pd.crosstab(clipDF[clipDF.reasonnotcompeted=='PDR: PATENT/DATA RIGHTS'].fiscal_year, clipDF[clipDF.reasonnotcompeted=='PDR: PATENT/DATA RIGHTS'].psc_simple, clipDF[clipDF.reasonnotcompeted=='PDR: PATENT/DATA RIGHTS'].dollarsobligated, aggfunc='count').plot(style='.-', **{'ms':12})
plt.gca().yaxis.set_major_formatter(generalFormatter)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos:'{:.0f}'.format(x)))
plt.legend(title='', frameon=True, shadow=True)
plt.grid(alpha = 0.6)
plt.xlabel('')
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\countserviceorproduct.pdf', bbox_inches='tight')

#==============================================================================
# mean of p_or_s by fiscal year and resulting line plot
#==============================================================================
pd.crosstab(clipDF[clipDF.reasonnotcompeted=='PDR: PATENT/DATA RIGHTS'].fiscal_year, clipDF[clipDF.reasonnotcompeted=='PDR: PATENT/DATA RIGHTS'].psc_simple, clipDF[clipDF.reasonnotcompeted=='PDR: PATENT/DATA RIGHTS'].dollarsobligated, aggfunc='mean').plot(style='.-', **{'ms':12})
plt.gca().yaxis.set_major_formatter(millionFormatter)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos:'{:.0f}'.format(x)))
plt.legend(title='', loc='upper left', ncol=2, frameon=True, shadow=True)
plt.xlabel('')
plt.grid(alpha = 0.6)
plt.savefig(r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Visualizations\USASpending\meanserviceorproduct.pdf', bbox_inches='tight')


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