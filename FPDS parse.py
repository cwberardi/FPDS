# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:15:37 2016

@author: Chris
"""

from lxml import objectify
from lxml import etree
import pandas as pd
import numpy as np
import string
from FPDS_test import get_xmlList
pd.set_option('display.float_format', lambda x:'%f'%x)

HDF5path = r'C:\Users\Chris\Documents\MIT\Dissertation\FPDS\Data\FPDS.h5'
#%%
def FPDSxmlparse(path, default=np.nan, tags = ['{http://www.fpdsng.com/FPDS}award', '{http://www.fpdsng.com/FPDS}count']):
    '''
    Function iterates through each tag in the xml document selecting only those in the tags variable.
       input: 
           path [string] - valid file path to xml archive data
           default[optional] - value returned if dependent path is not in child
           tags[list] - tag values used in iterparse
       return: pandas dataframe with each xml child as a row
    '''
    assert type(path)   ==str, 'path is a string'    
    assert type(tags)   ==list, 'list of tags to use in iterparse'
    
    def childhandlr(child, path, dtype, default):
        '''
        extracts relevent xml tags using the python getattr method, which returns the respective data if the specificied 
        dependenet path is present or returns a default response if not present.
        input:
            child [objectify.ObjectifiedElement] - child for which the decendent path is desired
            path [string] - decendent path
            dtype [string] - data type for xml tag [options: 'text' or 'pyval']
            default [optional] - value used to return if xml tag is not present
        return: xml string, float, or integer depending on specified data type
        '''
        assert type(path)   ==str, 'path is not a string'
        assert type(dtype)  ==str, 'dtype is not a string'
        #TODO: need to figure out why the below assertion is throwing an error
        #assert type(child)  ==objectify.ObjectifiedElement, 'Child is not an objectify.ObjectifiedElement'
        
        return getattr(getattr(child, path, default), dtype, default)     
        
    context = etree.iterparse(path, events=('end',), tag=tags)
    
    data = {} 
    for i, (_, element) in enumerate(context):
        kid = objectify.fromstring(etree.tostring(element))
        if kid.tag == '{http://www.fpdsng.com/FPDS}count':
            total = kid.total.pyval
            print('Total # of xml elements: {:,}'.format(total))
        else:
        # 'count' child does not contain valid contract data, consequently it is skipped
            if hasattr(kid.awardID, 'referencedIDVID'):
                piidIDV = kid.awardID.referencedIDVID.PIID.text
                modNumIDV = kid.awardID.referencedIDVID.modNumber.text
                piid = kid.awardID.awardContractID.PIID.text
                modNum = kid.awardID.awardContractID.modNumber.text
            else: 
                piid = kid.awardID.awardContractID.PIID.text
                modNum = kid.awardID.awardContractID.modNumber.text
                piidIDV = None
                modNumIDV = None
            
            data[i]={'modNum' :                 modNum,
                       'piid' :                 piid,
                       'piidIDV' :              piidIDV,
                       'modNumIDV':             modNumIDV,
                       'agencyID' :             kid.awardID.awardContractID.agencyID.text, 
                       'transNum':              kid.awardID.awardContractID.transactionNumber.text,
                       'comp_type':             childhandlr(kid.competition, 'extentCompeted', 'text', default),
                       'vendorName':            childhandlr(getattr(kid.vendor, 'vendorHeader', default), 'vendorName', 'text', default),
                       'vendorLocState' :       childhandlr(getattr(kid.vendor.vendorSiteDetails,'vendorLocation',default), 'state', 'text', default),
                       'vendorLocZip' :         childhandlr(getattr(kid.vendor.vendorSiteDetails, 'vendorLocation', default), 'ZIPCode', 'pyval', default),
                       'vendorLocCongDist':     childhandlr(getattr(kid.vendor.vendorSiteDetails, 'vendorLocation', default), 'congressionalDistrictCode', 'pyval', default),
                       'vendorCOSmallBusDeter': childhandlr(kid.vendor, 'contractingOfficerBusinessSizeDetermination', 'text', default),
                       'placeOfPerfState':      childhandlr(kid.placeOfPerformance.principalPlaceOfPerformance, 'stateCode','text', default),
                       'placeofPerfZip':        childhandlr(kid.placeOfPerformance, 'placeOfPerformanceZIPCode', 'pyval', default),
                       'placeofPerfCongDist' :  childhandlr(kid.placeOfPerformance, 'placeOfPerformanceCongressionalDistrict', 'pyval', default),
                       'vendorDUNS':            childhandlr(kid.vendor.vendorSiteDetails.vendorDUNSInformation,'DUNSNumber','pyval', default),
                       'solprocedures':         childhandlr(kid.competition, 'solicitationProcedures','text', default),
                       'descripOfContReqS':     childhandlr(kid.contractData, 'descriptionOfContractRequirement','text', default),
                       'typ_set_aside':         childhandlr(kid.competition, 'typeOfSetAside', 'text', default),
                       'statuteExcpToFairOp':   childhandlr(kid.competition, 'statutoryExceptionToFairOpportunity','text', default),
    #                   'evaluated_Preference': getattr(kid.competition, 'evaluated_Preference', default),  #Dead attribute, no values in any of Other DoD entries
                       'obligatedAmount':       childhandlr(kid.dollarValues, 'obligatedAmount', 'pyval',default),
                       'base/ExerOptValue':     childhandlr(kid.dollarValues, 'baseAndExercisedOptionsValue', 'pyval' ,default),
                       'base/AllOptionsValue':  childhandlr(kid.dollarValues, 'baseAndAllOptionsValue', 'pyval',default),
                       'numOfOffersReceived':   childhandlr(kid.competition, 'numberOfOffersReceived', 'pyval', default),
                       'fedBizOpps':            childhandlr(kid.competition,'fedBizOpps','text', default),
                       'signedDate':            childhandlr(kid.relevantContractDates, 'signedDate','text',default),
                       'effectiveDate':         childhandlr(kid.relevantContractDates,'effectiveDate','text',default),
                       'currentCompletionDate': childhandlr(kid.relevantContractDates, 'currentCompletionDate', 'text',default),
                       'ultimateCompletionDate':childhandlr(kid.relevantContractDates, 'ultimateCompletionDate','text', default),
                       'k_OfficeAgencyID':      childhandlr(kid.purchaserInformation, 'contractingOfficeAgencyID','text', default),
                       'k_OfficeID':            childhandlr(kid.purchaserInformation, 'contractingOfficeID','text',default),
                       'forProfit':             childhandlr(kid.vendor.vendorSiteDetails.vendorOrganizationFactors.profitStructure, 'isForProfitOrganization','text', default),
                       'fundingReqAgencyID':    childhandlr(kid.purchaserInformation, 'fundingRequestingAgencyID', 'pyval', default),
                       'fundingReqOfficeID':    childhandlr(kid.purchaserInformation, 'fundingRequestingOfficeID', 'pyval',default),
                       'multiYearContract':     childhandlr(kid.contractData, 'multiYearContract','text', default),
                       'GFE-GFP':               kid.contractData['GFE-GFP'].text,
                       'typeOfContractPricing': childhandlr(kid.contractData, 'typeOfContractPricing','text', default),
                       'contractActionType':    childhandlr(kid.contractData, 'contractActionType', 'text',default),
                       'costOrPricingData':     childhandlr(kid.contractData, 'costOrPricingData','text', default), 
                       'undefinitizedAction':   childhandlr(kid.contractData, 'undefinitizedAction','text',default),
                       'perfBasedServiceK':     childhandlr(kid.contractData, 'performanceBasedServiceContract','text', default),
                       'contractFinancing':     childhandlr(kid.contractData, 'contractFinancing', 'text', default),
                       'foreignFunding':        childhandlr(kid.purchaserInformation, 'foreignFunding','text', default),
                       'productOrServiceCode':  childhandlr(kid.productOrServiceInformation, 'productOrServiceCode', 'text',default),
                       'contractBundling':      childhandlr(kid.productOrServiceInformation, 'contractBundling','text', default),
                       'claimantProgramCode':   childhandlr(kid.productOrServiceInformation, 'claimantProgramCode','text', default),
                       'principalNAICSCode':    childhandlr(kid.productOrServiceInformation, 'principalNAICSCode','pyval', default),
                       'systemEquipmentCode':   childhandlr(kid.productOrServiceInformation, 'systemEquipmentCode','text', default),
                       'itCommercialItemCat':   childhandlr(kid.productOrServiceInformation, 'informationTechnologyCommercialItemCategory','text', default),
                       'reason_not_competed':   childhandlr(kid.competition, 'reasonNotCompeted','text', default)
                       }
              
            element.clear() 
            if float(i)%10000 == 0: print('Extracting XML -> %0.1f%% complete' %(100*float(i)/total))
              
    df= pd.DataFrame.from_dict(data, orient = 'index')
    
    # Convert date columns into datetime format
    for x in ['ultimateCompletionDate','signedDate', 'currentCompletionDate', 'effectiveDate']: df[x] = pd.to_datetime(df[x], errors = 'coerce')
        
    # Check to make sure length of dataframe matches children in root 
    assert len(df)==total, 'Length of dataframe does not match number of xml elements'
    # Check to make sure there are not entirely null columns in dataframe
    assert all(df.isnull().all())==False, 'one or more columns in this dataframe is/are entirely null'
    
    return df
    
#%%
def FPDSstoreh5(path, key, df):
    '''
    This function adds a dataframe to an existing HDF5store.  However, before adding it checks to see if the key is alrady in the store.  
    If so, returns assertion error
    
    input:
        path [string] - file path to valid HDF5store
        key [string] - key value to be used to in HDF5 store
        df [Object] - valid pandas dataframe object
        
    returns: None
    '''
    assert type(path)   ==str 
    assert type(key)    ==str
    assert type(df)     ==pd.DataFrame
    
    storeName = ''.join(['/', key])
    store = pd.HDFStore(path, complib= 'zlib')
    
    if storeName in store.keys():
        print('****Key '+ key + ' already in HDF5 Store***')
        rewrite = input('Delete old key/dataframe and add new key/dataframe? (y/n): ')
        if rewrite.lower() == 'y':
            del store[key]
            store[key] = df
    else:
        store[key] = df
        
    print(store)    
    store.close()
    
#%%

for x in get_xmlList(r'C:\Users\Chris\Downloads\FPDS Archive\Navy'):    
    key = x.split('\\')[-1].split('.')[0].lstrip(string.digits+'-').lower().replace('-','_')
    FPDSstoreh5(HDF5path, key, FPDSxmlparse(x))

#%%%
def makeCat(columns, df):
    for x in columns:
        df[x] = df[x].astype('category')
        
store = pd.HDFStore(HDF5path, mode='r')
afDF = pd.DataFrame()
for x in store.keys():
    afDF = pd.concat((afDF, store[x]), ignore_index = True)
store.close()

catCols = ['GFE-GFP', 'comp_type', 'contractActionType','contractBundling','contractFinancing','costOrPricingData', 'typ_set_aside', 'typeOfContractPricing','undefinitizedAction',
           'vendorCOSmallBusDeter','statuteExcpToFairOp', 'solprocedures', 'reason_not_competed', 'agencyID', 'claimantProgramCode', 'fedBizOpps','foreignFunding',
           'fundingReqAgencyID','fundingReqOfficeID', 'itCommercialItemCat', 'multiYearContract']

makeCat(catCols, afDF)
#%%
#df.isnull().sum().sort_values(ascending=False).div(len(df))