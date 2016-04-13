# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:15:37 2016

@author: Chris
"""

from lxml import objectify
import pandas as pd
import numpy as np
from FPDS_test import tagtest
pd.set_option('float_format',"%.2f")
#%%
path = r'C:\Users\Chris\Downloads\DOD-OTHER_DOD-DEPT-10012014TO09302015-Archive\OTHER_DOD_AGENCIES-AWARD.xml'
parsed = objectify.parse(path)
root = parsed.getroot()

tagtest(root, 2)
#%%
def FPDSxmlparse(root, default=np.nan):
    '''
    Function sorts through each child of the xml root, except for the 'count' child, and extracts relevent xml tags using the python getattr method, which 
    returns the respective data if the specificied dependenet path is present or returns a default response if not present.
       input: 
           root - must be list lxml.objectify.elements
           default - value returned if dependent path is not in child
       return: pandas dataframe with each xml child as a row
    '''
    data = {}
    default = default

    
    for i, kid in enumerate(root.getchildren()):
        # 'count' child does not contain valid contract data, consequently it is skipped
        if kid.tag == '{http://www.fpdsng.com/FPDS}count':
            continue
        else:
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
            
            data[i]={'modNum' : modNum,
                       'piid' : piid,
                       'piidIDV' : piidIDV,
                       'modNumIDV': modNumIDV,
                       'agencyID' : kid.awardID.awardContractID.agencyID.text, 
                       'transNum': kid.awardID.awardContractID.transactionNumber.text,
                       'comp_type': getattr(getattr(kid.competition, 'extentCompeted', default), 'text', default),
                       'vendorName': kid.vendor.vendorHeader.vendorName.text,
                       'vendorLocationState' : getattr(getattr(kid.vendor.vendorSiteDetails.vendorLocation, 'state', default), 'text', default),
                       'vendorLocationZip' : getattr(getattr(kid.vendor.vendorSiteDetails.vendorLocation, 'ZIPCode', default), 'pyval', default),
                       'vendorLocaitonCongDist' : getattr(getattr(kid.vendor.vendorSiteDetails.vendorLocation, 'congressionalDistrictCode', default), 'pyval', default),
                       'vendorCOSmallBussinesDetermination': kid.vendor.contractingOfficerBusinessSizeDetermination.text,
                       'placeOfPerformanceState': getattr(getattr(kid.placeOfPerformance.principalPlaceOfPerformance, 'stateCode', default),'text', default),
                       'placeofPerformanceZip': getattr(getattr(kid.placeOfPerformance, 'placeOfPerformanceZIPCode', default), 'pyval', default),
                       'placeofPerformanceCongDist' : getattr(getattr(kid.placeOfPerformance, 'placeOfPerformanceCongressionalDistrict', default),'pyval', default),
                       'vendorDUNS': kid.vendor.vendorSiteDetails.vendorDUNSInformation.DUNSNumber.pyval,
                       'solicitation_procedures': getattr(getattr(kid.competition, 'solicitationProcedures', default),'text', default),
                       'descriptionOfContractRequirement': getattr(getattr(kid.contractData, 'descriptionOfContractRequirement', default),'text', default),
                       'typ_set_aside': getattr(getattr(kid.competition, 'typeOfSetAside', default), 'text', default),
                       'statuteExcpToFairOp': getattr(getattr(kid.competition, 'statutoryExceptionToFairOpportunity', default),'text', default),
    #                   'evaluated_Preference': getattr(kid.competition, 'evaluated_Preference', default),  #Dead attribute, no values in any of Other DoD entries
                       'obligatedAmount': kid.dollarValues.obligatedAmount.pyval,
                       'baseAndExercisedOptionsValue': kid.dollarValues.baseAndExercisedOptionsValue.pyval,
                       'baseAndAllOptionsValue': kid.dollarValues.baseAndAllOptionsValue.pyval,
                       'numberOfOffersReceived': kid.competition.numberOfOffersReceived.pyval,
                       'fedBizOpps': getattr(getattr(kid.competition, 'fedBizOpps', default),'text', default),
                       'signedDate':kid.relevantContractDates.signedDate.text,
                       'effectiveDate': kid.relevantContractDates.effectiveDate.text,
                       'currentCompletionDate':kid.relevantContractDates.currentCompletionDate.text,
                       'ultimateCompletionDate':kid.relevantContractDates.ultimateCompletionDate.text,
                       'contractingOfficeAgencyID': kid.purchaserInformation.contractingOfficeAgencyID.text,
                       'contractingOfficeID': kid.purchaserInformation.contractingOfficeID.text,
                       'forProfit': getattr(getattr(kid.vendor.vendorSiteDetails.vendorOrganizationFactors.profitStructure, 'isForProfitOrganization', default),'text', default),
                       'fundingRequestingAgencyID': kid.purchaserInformation.fundingRequestingAgencyID.pyval,
                       'fundingRequestingOfficeID': kid.purchaserInformation.fundingRequestingOfficeID.pyval,
                       'multiYearContract': getattr(getattr(kid.contractData, 'multiYearContract', default),'text', default),
                       'GFE-GFP': kid.contractData['GFE-GFP'].text,
                       'typeOfContractPricing' : getattr(getattr(kid.contractData, 'typeOfContractPricing', default),'text', default),
                       'contractActionType': kid.contractData.contractActionType.text,
                       'costOrPricingData': getattr(getattr(kid.contractData, 'costOrPricingData', default), 'text', default), 
                       'undefinitizedAction': kid.contractData.undefinitizedAction.text,
                       'performanceBasedServiceContract': getattr(getattr(kid.contractData, 'performanceBasedServiceContract', default),'text', default),
                       'contractFinancing': getattr(getattr(kid.contractData, 'contractFinancing', default), 'text', default),
                       'foreignFunding': kid.purchaserInformation.foreignFunding.text,
                       'productOrServiceCode': kid.productOrServiceInformation.productOrServiceCode.text,
                       'contractBundling': getattr(getattr(kid.productOrServiceInformation, 'contractBundling', default),'text', default),
                       'claimantProgramCode': getattr(getattr(kid.productOrServiceInformation, 'claimantProgramCode', default), 'text', default),
                       'principalNAICSCode': getattr(getattr(kid.productOrServiceInformation, 'principalNAICSCode', default), 'pyval', default),
                       'systemEquipmentCode': getattr(getattr(kid.productOrServiceInformation, 'systemEquipmentCode', default),'text', default),
                       'itCommercialItemCategory': getattr(getattr(kid.productOrServiceInformation, 'informationTechnologyCommercialItemCategory', default), 'text', default),
                       'reason_not_competed': getattr(getattr(kid.competition, 'reasonNotCompeted', default), 'text', default)}
                       
    df= pd.DataFrame.from_dict(data, orient = 'index')
    
    # Convert date columns into datetime format
    for x in ['ultimateCompletionDate','signedDate', 'currentCompletionDate', 'effectiveDate']: df[x] = pd.to_datetime(df[x])
        
    # Check to make sure length of dataframe matches children in root 
    assert len(df)==len(root.getchildren())-1
    # Check to make sure there are not entirely null columns in dataframe
    assert all(df.isnull().all())==False
    
    return df
#%%
df = FPDSxmlparse(root)
df.isnull().sum().sort_values(ascending=False).div(len(df))
