SELECT  Id
      , Anvil__AccountNumber__c 
      , Anvil__County__c
      , ParentId
      , OwnerId
      , Anvil__CallFrequency__c
      , Anvil__Call_Status__c
      , Anvil__Engagement_Level__c
      , Anvil__Org_Type__c
      , Anvil__Org_Sub_Type__c
      , BillingPostalCode
FROM Account 
WHERE OwnerId != '0058Z0000088DPsQAM' --Ignore system user