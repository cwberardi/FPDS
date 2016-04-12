# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 11:15:37 2016

@author: Chris
"""

from lxml import objectify
import pandas as pd
from FPDS_test import tagtest, lentest
#%%
path = r'C:\Users\Chris\Downloads\DOD-OTHER_DOD-DEPT-10012014TO09302015-Archive\OTHER_DOD_AGENCIES-AWARD.xml'
parsed = objectify.parse(path)
root = parsed.getroot()

tagtest(root, 2)
#%%
data = {}
default = None

for i, kid in enumerate(root.getchildren()):
    if kid.tag == '{http://www.fpdsng.com/FPDS}count':
        continue
    else:
        if hasattr(kid.awardID, 'referencedIDVID'):
            piid = kid.awardID.referencedIDVID.PIID
            modNum = kid.awardID.referencedIDVID.modNumber
        else: 
            piid = kid.awardID.awardContractID.PIID
            modNum = kid.awardID.awardContractID.modNumber
        
        data[i]={'modNum' : modNum,
                   'piid' : piid,
                   'agencyID' : kid.awardID.awardContractID.agencyID.text, 
                   'transNum': kid.awardID.awardContractID.transactionNumber.text,
                   'comp_type': getattr(kid.competition, 'extentCompeted', default),
                   'vendorName': kid.vendor.vendorHeader.vendorName.text,
                   'vendorLocationState' :getattr(kid.vendor.vendorSiteDetails.vendorLocation, 'state', default),
                   'vendorLocationZip' :getattr(kid.vendor.vendorSiteDetails.vendorLocation, 'ZIPCode', default),
                   'vendorLocaitonCongDist' : getattr(kid.vendor.vendorSiteDetails.vendorLocation, 'congressionalDistrictCode', default),
                   'vendorCOSmallBussinesDetermination': kid.vendor.contractingOfficerBusinessSizeDetermination.text,
                   'placeOfPerformanceState': getattr(kid.placeOfPerformance.principalPlaceOfPerformance, 'stateCode', default),
                   'placeofPerformanceZip': getattr(kid.placeOfPerformance, 'placeOfPerformanceZIPCode', default),
                   'placeofPerformanceCongDist' : getattr(kid.placeOfPerformance, 'placeOfPerformanceCongressionalDistrict', default),
                   'vendorDUNS': kid.vendor.vendorSiteDetails.vendorDUNSInformation.DUNSNumber.text,
                   'solicitation_procedures': getattr(kid.competition, 'solicitationProcedures', default),
                   'typ_set_aside': getattr(kid.competition, 'typeOfSetAside', default),
                   'evaluated_Preference': getattr(kid.competition, 'evaluated_Preference', default),
                   'obligatedAmount': kid.dollarValues.obligatedAmount.text,
                   'baseAndExercisedOptionsValue': kid.dollarValues.baseAndExercisedOptionsValue.text,
                   'baseAndAllOptionsValue': kid.dollarValues.baseAndAllOptionsValue.text,
                   'numberOfOffersReceived': kid.competition.numberOfOffersReceived.text,
                   'fedBizOpps': getattr(kid.competition, 'fedBizOpps', default),
                   'signedDate':kid.relevantContractDates.signedDate.text,
                   'effectiveDate': kid.relevantContractDates.effectiveDate.text,
                   'currentCompletionDate':kid.relevantContractDates.currentCompletionDate.text,
                   'ultimateCompletionDate':kid.relevantContractDates.ultimateCompletionDate.text,
                   'contractingOfficeAgencyID': kid.purchaserInformation.contractingOfficeAgencyID.text,
                   'contractingOfficeID': kid.purchaserInformation.contractingOfficeID.text,
                   'fundingRequestingAgencyID': kid.purchaserInformation.fundingRequestingAgencyID.text,
                   'fundingRequestingOfficeID': kid.purchaserInformation.fundingRequestingOfficeID.text,
                   'multiYearContract': getattr(kid.contractData, 'multiYearContract', default),
                   'GFE-GFP': kid.contractData['GFE-GFP'].text,
                   'typeOfContractPricing' : getattr(kid.contractData, 'typeOfContractPricing', default),
                   'contractActionType': kid.contractData.contractActionType.text,
                   'costOrPricingData': getattr(kid.contractData, 'costOrPricingData', default), 
                   'undefinitizedAction': kid.contractData.undefinitizedAction.text,
                   'performanceBasedServiceContract': getattr(kid.contractData, 'performanceBasedServiceContract', default),
                   'contractFinancing': getattr(kid.contractData, 'contractFinancing', default),
                   'foreignFunding': kid.purchaserInformation.foreignFunding.text,
                   'productOrServiceCode': kid.productOrServiceInformation.productOrServiceCode.text,
#                   'contractBundling': kid.productOrServiceInformation.contractBundling.text,
                   'claimantProgramCode': getattr(kid.productOrServiceInformation, 'claimantProgramCode', default),
                   'principalNAICSCode': getattr(kid.productOrServiceInformation, 'principalNAICSCode', default),
                   'systemEquipmentCode': getattr(kid.productOrServiceInformation, 'systemEquipmentCode', default),
                   'itCommercialItemCategory': getattr(kid.productOrServiceInformation, 'informationTechnologyCommercialItemCategory', default),
                   'reason_not_competed': getattr(kid.competition, 'reasonNotCompeted', default)}
#%%
df = pd.DataFrame.from_dict(data, orient = 'index')  
lentest(df, root)                         