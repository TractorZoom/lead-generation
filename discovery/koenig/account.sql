SELECT  Id
      , Anvil__AccountNumber__c 
      , County__c
      , Anvil__AORCustomer__c
      , Anvil__PrimaryStoreLocation__c
      , ParentId
      , OwnerId
      , Anvil__Technology_Rep__c
      , Anvil__Customer_Type__c
      , Equipment__c
      , Anvil__Competitive_Owner__c
      , Anvil__CallFrequency__c
      , Anvil__Call_Status__c
      , Anvil__Trade_Type__c
      , Anvil__Engagement_Level__c
      , BillingPostalCode
FROM Account 
WHERE OwnerId != '0055f000000wHZ1AAM'  -- Exclude Anvil Admin as owner