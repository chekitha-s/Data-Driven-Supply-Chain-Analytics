# Data Driven Supply Chain Analytics
The dataset for this project originates from a rich supply chain and sales context, aiming to explore various facets of customer behavior, product performance, and logistical efficiency. It includes a wealth of data entries and features that span sales information, shipping details, and product categories. This project explores how data mining techniques can improve supply chain efficiency and customer satisfaction. Using a dataset from Kaggle’s Smart Supply Chain for Big Data Analysis the analysis covers customer behavior, product performance, and delivery logistics.

The primary research question for this project is: How can data-driven insights enhance operational efficiency and customer satisfaction in supply chain management?

### SMART QUESTION:
1. What factors most significantly impact delivery times, and how can we mitigate the risk of late deliveries?
2. Can customer purchasing patterns be predicted to enhance inventory management and marketing strategies?
3. How does the product category relate to sales performance, and can this insight inform inventory stocking?
4. What is the relationship between shipping mode choice and customer satisfaction or sales benefit?
5. Is there any significant difference in average order profit between regions?

### MODELLING METHODS:
1. Descriptive Analysis: Summary statistics, correlations, and visualizations to identify key drivers of performance.
2. Customer Segmentation: K-Means and Hierarchical Clustering to group customers by purchasing patterns.
3. Forecasting: Linear Regression and ARIMA for predicting sales and shipping trends.
4. Delivery Risk Prediction: Logistic Regression, Decision Trees, Random Forest, SVM for predicting late deliveries.
5. Advanced Predictive Modeling: GBM, XGBoost for higher accuracy on classification and regression tasks.
6. Optimization: Linear Programming for supply chain optimization and resource allocation.
   
### Product Trend Analysis and Sales Trends Overtime
#### Product Trends:
1. ANOVA tests to measure variable impacts on sales.
2. Visualization of sales across product categories.
3. Grouping categories to identify highest performing segments.
#### Sales Over Time:
1. Peak and low sales months identified.
2. Tracking categories that remain in demand despite downturns, supporting inventory planning.

### Applications
Inventory Optimization – stocking the right products at the right time.
Delivery Risk Mitigation – proactive management of late shipments.
Customer Segmentation – enabling targeted marketing strategies.
Regional Analysis – tailoring operations by geography.

### Conclusions
Delivery Delays: Shipping mode and region are key factors in late deliveries. Air shipping reduces delays significantly compared to standard or road transport.
Customer Segmentation: Distinct customer groups emerged — some highly price-sensitive, others loyal to premium categories. This segmentation enables tailored marketing strategies.
Sales Drivers: Technology and office supply categories consistently outperform others, while seasonal trends (e.g., end-of-year spikes) play a major role in sales volume.
Inventory Insights: Certain categories maintain steady demand regardless of market downturns, suggesting they should always remain in stock for stability.
Profitability: Regional differences in order profit were identified, indicating that operations can be optimized by focusing on higher-margin regions.
Predictive Power: Machine learning models (Random Forest, XGBoost) provided reliable predictions for delivery risk and sales trends, demonstrating the value of advanced analytics.
Optimization Impact: Linear programming simulations highlighted opportunities to minimize delivery costs while maintaining service quality.
   
### Source:
The dataset for this project is obtained from Kaggle.
https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
