SELECT Id
     , Anvil__DealerStockUnit__c
     , Anvil__Invoice_Number__c
     , Anvil__Account_Number__c	
     , Anvil__Billing_Date__c
     , Anvil__Customer_Number__c
     , Anvil__Branch__c
     , Anvil__Amount__c
     , Anvil__Description__c
FROM Anvil__Equipment_History__c
WHERE Anvil__Complete_Goods_SubType__c = 'S'