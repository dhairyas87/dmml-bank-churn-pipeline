-- === Sample SQL Queries === --

-- Preview data
SELECT * FROM transformed_churn LIMIT 10;

-- Customer distribution by AgeGroup
SELECT AgeGroup, COUNT(*) AS Customers FROM transformed_churn GROUP BY AgeGroup;

-- Balance-Salary ratio by CreditScoreBucket
SELECT CreditScoreBucket, AVG(BalanceSalaryRatio) FROM transformed_churn GROUP BY CreditScoreBucket;

-- Average CreditScore by Active Member
SELECT IsActiveMember, AVG(CreditScore) FROM transformed_churn GROUP BY IsActiveMember;

-- Churned customers by Geography
SELECT Geography, SUM(Exited) AS Churned_Customers FROM transformed_churn GROUP BY Geography;

