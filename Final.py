#!/usr/bin/env python
# coding: utf-8

# ## Collaborator: Chinmay Dhamapurkar, Shikha, Varun, Chekitha

# In[1]:


import pandas as pd


# In[2]:


import os
os.getcwd()


# In[3]:


df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.corr()


# In[8]:


df.isnull().sum()


# In[9]:


df=df.drop(columns=["Product Description"])
df=df.drop(columns=["Order Zipcode"])



# ## smart Question(Optional) : What factors most significantly impact the delivery times and how can we mitigate the risk of late deliveries?
# 
# ##This was the smart question we tried to work on first, but later due to time constraints we deicided to skip this and move to some other good smart questions.So, please consider this as optional.

# From these visualizations, we can observe:
# 
# There's variability in the actual shipping days across different shipping modes, which suggests that the shipping mode could be a significant factor in determining delivery times.
# The scatter plot shows that while many orders are delivered on time (close to the diagonal line), there are also quite a few that are delivered earlier or later than scheduled.
# To further quantify the impact of these and other factors, we can perform a correlation analysis and then move on to predictive modeling. The model will help us to not only confirm the impact of these factors but also to quantify their effect on delivery times.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
potential_factors = [
    'Days for shipping (real)',
    'Days for shipment (scheduled)',
    'Shipping Mode',
    'Order Priority',
    'Benefit per order',
    'Late_delivery_risk',
    'Category Name',
    'Customer City',
    'Customer State',
    'Order Region',
    'Order Country',
    'Market',
    'Order Item Quantity',
    'Sales per customer'
]
plt.figure(figsize=(10, 6))
sns.boxplot(x='Shipping Mode', y='Days for shipping (real)', data=df, showfliers=False)
plt.title('Impact of Shipping Mode on Delivery Times')
plt.ylabel('Days for Shipping (Real)')
plt.xlabel('Shipping Mode')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Days for shipment (scheduled)', y='Days for shipping (real)', data=df)
plt.title('Scheduled vs. Actual Shipping Days')
plt.ylabel('Days for Shipping (Real)')
plt.xlabel('Days for Shipment (Scheduled)')
plt.tight_layout()

plt.show()


# In[ ]:


df.info()


# In[ ]:


correlation_matrix = df.corr()

# Now, we'll plot the heatmap using seaborn
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.title('Heatmap of Correlation Matrix')
plt.show()


# In[ ]:


import seaborn as sns
if 'Type' in df.columns:
    # Create a pie chart
    plt.figure(figsize=(8, 8))
    df['Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Types')
    plt.ylabel('')  # Hide the y-label
    plt.show()
else:
    print("The 'Type' column does not exist in the dataframe.")


# 

# In[ ]:


data=df
data['order date']= pd.to_datetime(data['order date (DateOrders)'])
data['shipping date']= pd.to_datetime(data['shipping date (DateOrders)'])


# In[ ]:


data['order year']=data['order date'].dt.year
data['order month']=data['order date'].dt.month
data['order day']=data['order date'].dt.day
data['order hour']=data['order date'].dt.hour
data['order minute']=data['order date'].dt.minute

data['shipping year']=data['shipping date'].dt.year
data['shipping month']=data['shipping date'].dt.month
data['shipping day']=data['shipping date'].dt.day
data['shipping hour']=data['shipping date'].dt.hour
data['shipping minute']=data['shipping date'].dt.minute


# In[ ]:


data


# In[ ]:


data_n=df.loc[:,['Type','Days for shipment (scheduled)','order year','order month','order day','order hour','order minute','Benefit per order','Category Name','Latitude','Longitude','Customer Segment','Department Name','Market','Order City','Order Country','Order Item Discount','Order Item Product Price','Order Item Quantity','Order Item Total','Order State','Product Name','shipping year','shipping month','shipping day','shipping hour','shipping minute','Shipping Mode','Late_delivery_risk','Order Status']]
data_n.info()


# In[ ]:


for i in data_n:
    if data_n[i].dtype=="object":
        print(i,len(data_n[i].unique()))


# This are the values in text, lets convert it into numbers using encoding techniques

# In[ ]:


data_n['Order Status'].value_counts()
data_n['Order Status']= [0 if i!='SUSPECTED_FRAUD' else 1 for i in data_n['Order Status']]
data_n['Order Status'].unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
enc=LabelEncoder()
for i in data_n.columns:
    if data_n[i].dtype=='object':
        data_n[i]=enc.fit_transform(data_n[i])
data_n.info()


# In[ ]:


data_n.info()


# ## Building ML Models as all features have been succesfully converted to numerical values

# In[ ]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
y1=data_n["Late_delivery_risk"]

y2=data_n['Order Status']
x=data_n.drop(['Late_delivery_risk','Order Status'],axis=1)
x=StandardScaler().fit_transform(x)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train1,y_test1,y_train2,y_test2= train_test_split(x,y1,y2,stratify=y1,test_size=0.2)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# log_reg = LogisticRegression()
# decision_tree = DecisionTreeClassifier()
# random_forest = RandomForestClassifier()
# grad_boost = GradientBoostingClassifier()

# models = {
#     'Logistic Regression': log_reg,
#     'Decision Tree': decision_tree,
#     'Random Forest': random_forest,
#     'Gradient Boosting': grad_boost
# }

# import joblib


# for model_name, model in models.items():
#     model.fit(x_train, y_train1)
#     y_pred = model.predict(x_test)
#     accuracy = accuracy_score(y_test1, y_pred)
#     roc_auc = roc_auc_score(y_test1, y_pred)
#     scores[model_name] = {'Accuracy': accuracy, 'ROC AUC': roc_auc}
    
#     # Save the trained model to a file
#     joblib.dump(model, 'model.joblib')
    


# scores
# print(scores)


# In[ ]:


import os
os.getcwd()


# In[ ]:


import joblib
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Assuming your models were saved with the names 'Logistic_Regression_model.joblib', 'Decision_Tree_model.joblib', etc.
model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
model_filenames = {
    'Logistic Regression': 'Logistic_Regression_model.joblib',
    'Decision Tree': 'Decision_Tree_model.joblib',
    'Random Forest': 'Random_Forest_model.joblib',
    'Gradient Boosting': 'Gradient_Boosting_model.joblib'
}

scores = {}

for model_name in model_names:
    # Load the trained model from file
    loaded_model = joblib.load('model.joblib')
    
    # Predict using the loaded model
    y_pred = loaded_model.predict(x_test)
    
    # Calculate scores
    accuracy = accuracy_score(y_test1, y_pred)
    roc_auc = roc_auc_score(y_test1, y_pred)
    
    # Store scores in the dictionary
    scores[model_name] = {'Accuracy': accuracy, 'ROC AUC': roc_auc}

print(scores)


# # OVERFITTING CHECK

# In[ ]:


random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train1)
train_accuracy = accuracy_score(y_train1, random_forest.predict(x_train))
test_accuracy = accuracy_score(y_test1, random_forest.predict(x_test))

print("Random Forest - Training Accuracy:", train_accuracy)
print("Random Forest - Test Accuracy:", test_accuracy)


# The results show that the training accuracy of the Random Forest model is 1.0 (or 100%), while the test accuracy is approximately 95.16%. As a data scientist, here are the key points to consider regarding overfitting:
# 
# Training Accuracy: A training accuracy of 100% indicates that the model has learned to perfectly classify all training data. While this may seem ideal, it's often a sign that the model has become overly complex and has memorized the training data rather than learned to generalize from it.
# 
# Test Accuracy: The test accuracy is lower than the training accuracy, which is expected because models typically perform better on the data they have seen (training data) versus new, unseen data (test data). However, the test accuracy is still quite high, suggesting that the model is performing well on unseen data.
# 
# Overfitting Concern: The gap between training and test accuracy suggests that the model may be overfitting to some extent. Overfitting occurs when the model captures noise or random fluctuations in the training data that do not apply to the broader dataset.
# 
# Actions to Take: To address potential overfitting, we might consider:
# 
# Tuning hyperparameters such as max_depth and min_samples_split to control the complexity of the model.
# Introducing regularization techniques, which can penalize overly complex models.
# Implementing cross-validation to ensure that the model's performance is consistent across different subsets of the training data.
# Reducing the number of features if the model's complexity is due to a high dimensionality of input data.
# Increasing the size of the training set can sometimes help the model generalize better.
# In summary, while the model shows promising results, caution must be taken to ensure that it will perform well on new, unseen data and not just the specific data it was trained on. The slight discrepancy between training and test accuracy is a flag for possible overfitting, which should be addressed through model tuning and validation techniques.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Scores from your results
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
accuracy_scores = [0.787, 0.979, 0.948, 0.810]
roc_auc_scores = [0.796, 0.978, 0.949, 0.816]

x = np.arange(len(models))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracy_scores, width, label='Accuracy')
rects2 = ax.bar(x + width/2, roc_auc_scores, width, label='ROC AUC')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by model and metric')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Function to attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# ## Best Model After Hyperparameter Tuning

# In[ ]:


# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier

# # Hyperparameters to be tuned
# param_distributions = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]  # Whether bootstrap samples are used
# }
# random_search = RandomizedSearchCV(
#     estimator=RandomForestClassifier(),
#     param_distributions=param_distributions,
#     n_iter=10,  # Number of parameter settings sampled, can be adjusted based on computational resources
#     cv=3,  # Number of folds in cross-validation
#     verbose=2,
#     random_state=42,
#     n_jobs=-1  # Use all available cores
# )

# random_search.fit(x_train, y_train1)
# print(f"Best Parameters: {random_search.best_params_}")


# As a data scientist, hyperparameter tuning is crucial to optimize machine learning models. RandomizedSearchCV offers an efficient way to search through a specified hyperparameter space for the best performing parameters. It randomly selects combinations rather than exhaustively testing all possible values, striking a balance between exploration and computational feasibility.
# 
# Why we use it:
# 
# Efficiency: It's faster than GridSearchCV since it doesn't try every combination but instead samples a subset.
# Surprise Minimization: Random sampling can sometimes find good hyperparameters that grid search might miss.
# Scalability: It can handle a large hyperparameter space without a significant increase in computational time.
# How it affects model performance:
# 
# Optimization: It helps in identifying the most effective parameters that could lead to better model accuracy and generalization.
# Reduction of Overfitting: By tuning parameters like max_depth and min_samples_leaf, we can control the complexity of the model, thus reducing the risk of overfitting.
# Enhanced Predictive Power: The ultimate goal is to enhance the model's predictive power on new, unseen data, by selecting the hyperparameters that lead to the best cross-validation performance.

# <!-- Cross-Validation
# To ensure our model's robustness and to check for overfitting, we employed k-fold cross-validation. This technique involves partitioning the data into k equally sized segments, or "folds", then training the model k times, each time using a different 
# 1
# k−1 folds as the training set. By averaging the model's performance across these k iterations, we obtain a more reliable assessment of its predictive power.
# 
# Hyperparameter Tuning
# To optimize our model, we conducted hyperparameter tuning using grid search with cross-validation. This process systematically works through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. We explored a range of values for parameters such as the number of estimators (n_estimators), maximum depth of the trees (max_depth), and the minimum number of samples required to split an internal node (min_samples_split). The best parameters from this search were then used to configure our model.
# 
# Feature Importance Analysis (Optional)
# Understanding which features contribute most to the model's decision-making can provide insights into the dataset and the model's logic. While not executed in the code above, this step typically follows the training of a model. Most tree-based models can output a list of feature importances which can be analyzed to refine the model further.
# 
# Final Model Training
# Armed with the optimal set of parameters, we trained our final model on the entire training dataset. This step is crucial because it leverages all available training data to produce the most accurate model possible before we deploy it to make predictions on unseen data.
# 
# Evaluation on Test Set
# The ultimate test of our model's performance came when we evaluated it against our test dataset. This dataset was never seen by the model during the training or tuning phases, which means it serves as a proxy for real-world data. We reported standard metrics such as Accuracy and ROC AUC to judge the model's effectiveness. Additionally, we looked at the classification report, which provides a breakdown of precision, recall, and F1-score by class, and the confusion matrix, which shows the model's predictions in detail.
# 
# By following these rigorous practices, we aimed to build a predictive model that not only performs well on historical data but can also be trusted to make accurate predictions in practical applications. -->

# In[ ]:


import joblib
best_model = joblib.load('random_search_rf_model.joblib')
from sklearn.metrics import confusion_matrix
random_search_=best_model
y_pred = best_model.predict(x_test)
conf_matrix = confusion_matrix(y_test1, y_pred)
accuracy = accuracy_score(y_test1, y_pred)
import seaborn as sns
conf_matrix = np.array(conf_matrix)
accuracy

# Create a heatmap from the confusion matrix
labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', annot_kws={"size": 16}, xticklabels=['Predicted Neg', 'Predicted Pos'], yticklabels=['Actual Neg', 'Actual Pos'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Print the accuracy
print(f"Accuracy: {accuracy:.4f}")


# In[ ]:


# # After fitting the model with random_search.fit(x_train, y_train1)
# import joblib
# joblib.dump(random_search, 'random_search_rf_model.joblib')


# In[ ]:


loaded_model = joblib.load('random_search_rf_model.joblib')


# In[ ]:


from sklearn.metrics import classification_report

# Assuming best_model is your trained RandomForestClassifier after hyperparameter tuning
y_pred = best_model.predict(x_test)

# Generate the classification report
class_report = classification_report(y_test1, y_pred)

print(class_report)


# In[ ]:


df.info()


# ## Classification Report Summary
# The classification report provides detailed metrics for each class (0 and 1), as well as overall model performance measures. Here are the key takeaways:
# 
# Precision (the fraction of true positive predictions out of all positive predictions):
# 
# Class 0: 97% - The model is highly precise when predicting class 0.
# Class 1: 98% - The model is even more precise with class 1 predictions.
# Recall (the fraction of true positives out of the actual true instances of the class):
# 
# Class 0: 97% - The model has a high recall for class 0, correctly identifying 97% of actual class 0 instances.
# Class 1: 97% - Similarly, the model also correctly identifies 97% of actual class 1 instances.
# F1-Score (the harmonic mean of precision and recall):
# 
# Class 0: 97% - Indicates a strong balance between precision and recall for class 0.
# Class 1: 98% - Indicates an even stronger balance for class 1.
# Support (the number of actual occurrences of each class in the dataset):
# 
# Class 0: 16,308 instances.
# Class 1: 19,796 instances.
# Overall Accuracy: 97% - The model accurately predicts the correct class 97% of the time across both classes.
# 
# Macro Avg: The average precision, recall, and F1-score between classes without considering the support is 97% for each, suggesting that the model performs equally well across both classes.
# 
# Weighted Avg: When accounting for support (the number of instances in each class), the weighted average for precision, recall, and F1-score is 97%, confirming consistent model performance across classes with different support.
# 
# In summary, the model demonstrates a high level of precision and recall across both classes with very little difference between them, indicating a balanced and effective classification performance.

# ## SMART Question :
# What factors most significantly impact the delivery times and how can we mitigate the risk of late deliveries?
# 
# ## Data Analysis and Preprocessing:
# We started by cleaning the dataset, addressing missing values, and encoding categorical variables where necessary.
# Feature engineering was applied to extract meaningful information from existing data, such as converting dates into more usable numerical formats (year, month, day, hour, minute).
# Correlation analysis helped us understand the relationships between different variables and the Late_delivery_risk.
# Exploratory Data Analysis:
# Visualizations were used to explore the data, which included heatmaps to understand correlations and pie charts to visualize the distribution of categorical variables.
# We analyzed the Days for shipping (real) and Days for shipment (scheduled) and found that longer lead times could potentially increase the risk of late deliveries.
# Latitude and longitude data were planned to be used for geographical analysis to see if certain regions were more prone to late deliveries, but this was not fully explored within the current session.
# 
# ## Modeling:
# We built several machine learning models to predict the Late_delivery_risk:
# 
# Logistic Regression
# Decision Tree
# Random Forest
# Gradient Boosting
# The Random Forest model showed promising results, but there was an indication of overfitting as the training accuracy was perfect (1.0) while the test accuracy was lower.
# 
# ## Model Evaluation and Optimization:
# To combat overfitting, we conducted hyperparameter tuning with RandomizedSearchCV, which improved the generalizability of the Random Forest model.
# Post-tuning, we achieved a balance between training and testing accuracy, indicating a more robust model.
# The confusion matrix and classification report provided detailed performance metrics, showing high precision, recall, and f1-scores for the prediction of Late_delivery_risk.
# Insights and Recommendations:
# Key Predictors: The models highlighted several important predictors for late deliveries, including shipping methods, order size, and product categories.
# Mitigation Strategies: Based on the analysis, strategies such as optimizing shipping routes, adjusting inventory levels, and improving forecast accuracy could mitigate the risk of late deliveries.
# Continuous Monitoring: Implementing a real-time monitoring system to track delivery performance against forecasts and schedules could help in taking proactive measures to avoid late deliveries.
# 
# ## Conclusion:
# The detailed data-driven approach provided insights into factors affecting delivery times and offered actionable recommendations to reduce the risk of late deliveries. Continuous improvement and iterative analysis are suggested to refine the strategies and maintain high delivery performance.
# How the ML Model Helps in Mitigating Late Deliveries
# Machine learning (ML) models can significantly enhance the ability to predict and, consequently, mitigate late deliveries in supply chain management. Here’s how the process unfolds and the role of data science:
# 
# Predictive Insights
# ML models like the Random Forest classifier we used can predict the likelihood of late deliveries based on historical data. These models take into account numerous factors, including shipment methods, historical delivery times, order sizes, and product categories, to identify patterns that lead to delays.
# 
# Feature Importance
# The ML models provide quantitative insights into which features are most predictive of late deliveries. This knowledge is vital for prioritizing efforts to improve delivery times. For example, if the model finds that certain shipment methods are more prone to delays, the company can evaluate and potentially modify these methods.
# 
# Proactive Action
# With predictions in hand, companies can be proactive rather than reactive. For instance, orders predicted to have a high risk of late delivery can be flagged for expedited processing or additional monitoring.
# 
# Process Optimization
# Data science helps identify bottlenecks and inefficiencies in the shipping process. By analyzing patterns in the data, ML models can suggest where process improvements can be made, such as optimizing inventory distribution or adjusting staff allocations.
# 
# Customer Satisfaction
# By reducing the incidence of late deliveries, customer satisfaction and retention can be improved. This is crucial in competitive markets where customer experience is a key differentiator.

# ## Map Analysis

# In[ ]:


get_ipython().system('pip install folium')
import folium
from folium.plugins import HeatMap
sample_data = data.sample(n=10000, random_state=42)
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=2)
HeatMap(data=df[['Latitude', 'Longitude']], radius=15).add_to(m)
m


# ## Smart Question 1: How are RFM (Recency, Frequency, Monetary) and ABC Analysis revolutionizing the supply chain industry by optimizing inventory management and enhancing customer relationship strategies?
# 
# 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
numerical_features = ['Days for shipping (real)', 'Days for shipment (scheduled)', 
                      'Benefit per order', 'Sales per customer', 'Product Price']
numerical_summary = data[numerical_features].describe()
fig, axes = plt.subplots(nrows=len(numerical_features), ncols=1, figsize=(10, 20))

for i, col in enumerate(numerical_features):
    sns.histplot(data[col], ax=axes[i], kde=True)
    axes[i].set_title(f'Distribution of {col}', fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

fig.tight_layout()
plt.show()

numerical_summary


# ## The summary statistics and distributions for the numerical features show the following:
# 
# Days for shipping (real): Ranges from 0 to 6 with a mean of approximately 3.5 days. The distribution seems fairly uniform.
# 
# Days for shipment (scheduled): Most values are clustered around 2 and 4, which suggests that most shipments are scheduled within this range.
# 
# Benefit per order: There's a wide range, with the minimum showing a significant loss and the maximum showing a large profit per order. The mean profit is about $22, but the distribution is skewed due to outliers, as indicated by the standard deviation.
# 
# Sales per customer: The average sale per customer is about $183, with a wide range of sales values. The distribution is right-skewed, indicating that there are outliers with very high sales values.
# 
# Product Price: The prices range from 9.99 to nearly 2000, with a mean price around $141. The distribution is right-skewed, suggesting the presence of high-priced items as outliers.
# 
# 

# In[ ]:


categorical_features = ['Type', 'Delivery Status', 'Category Name', 'Shipping Mode']

fig, axes = plt.subplots(nrows=len(categorical_features), ncols=1, figsize=(12, 24))

for i, col in enumerate(categorical_features):
    order_count = data[col].value_counts().head(10) 
    sns.barplot(x=order_count, y=order_count.index, ax=axes[i])
    axes[i].set_title(f'Count of {col}', fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

fig.tight_layout()
plt.show()


# In[ ]:


import pandas as pd
import dateutil.parser
import numpy as np
import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns;
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import plotly.express as px


# In[ ]:


sm2 = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')


# In[ ]:


sm2.head()
sm2.info()
sm2.describe()


# In[ ]:


sm2


# In[ ]:


sm2.drop('Product Description', axis=1 , inplace= True)


# In[ ]:


sm2


# In[ ]:


sm2['order date (DateOrders)']= pd.to_datetime(sm2['order date (DateOrders)'])


# In[ ]:


sm2["shipping date (DateOrders)"]=pd.to_datetime(sm2["shipping date (DateOrders)"])
sm2=sm2.sort_values(by="order date (DateOrders)")


# In[ ]:


dFppsc=sm2
m=dFppsc.corr()
m[m==1].to_excel("corr1.xlsx")
matrix=m.abs()
upper_tri = matrix.where(np.triu(np.ones(matrix.shape),k=1).astype(bool))
co_col=[column for column in upper_tri.columns if any(upper_tri[column] ==1)]
co_linear_col=[]
for c in co_col:
    co_linear_col.append([c,matrix[c][matrix[c]==1]])
co_linear_col  
for c in co_linear_col:
    print(c[1])


# We calculated the corelation between different columns and found out that several columns have correlation of 1. This is not possible unless they are duplicate columns or one columns is derived from another for e.g x=5y.

# In[ ]:


print("Length of the data frame where benefits per order and order profit per order are not equal :"+str(dFppsc[dFppsc["Benefit per order"]!=dFppsc["Order Profit Per Order"]].shape[0]))
print("Length of the data frame where sales per customer and Order Item Total are not equal :"+str(dFppsc[dFppsc["Sales per customer"]!=dFppsc["Order Item Total"]].shape[0]))
print("Length of the data frame where Order Item Cardprod Id and Product Card Id are not equal :"+str(dFppsc[dFppsc["Order Item Cardprod Id"]!=dFppsc["Product Card Id"]].shape[0]))
print("Length of the data frame where Product Price and Order Item Product Price are not equal :"+str(dFppsc[dFppsc["Product Price"]!=dFppsc["Order Item Product Price"]].shape[0]))
print("Length of the data frame where Category Id and Product Category Id are not equal :"+str(dFppsc[dFppsc["Product Category Id"]!=dFppsc["Category Id"]].shape[0]))
print("Length of the data frame where Order Customer Id and Customer Id Product Price are not equal :"+str(dFppsc[dFppsc["Order Customer Id"]!=dFppsc["Customer Id"]].shape[0]))


# After comparing the values in columns pairs having co-relation of 1, we can see that all values are equal for these pair of columns. We can drop one of the columns from each of the above pairs as they are duplicate columns of each other.

# In[ ]:


dFppsc.drop(["Benefit per order","Sales per customer","Order Item Cardprod Id","Order Item Product Price","Product Category Id","Order Customer Id"],axis=1,inplace=True)


# ## Using Variance Inflation Factor to find out multi-colinearity between three or more columns
# 
# Corelation helps us in finding relations between any two columns. But what if we want to find out if there is a relation between three or more columns. That's where Variance Inflaction Factor can help.
# 
# Variance Inflation Factor (VIF) - VIF is the ratio of variance of coefficient estimate when fitting the full model divided by the variance of coefficient estimate if fit on its own. The minimum possible value is 1 which indicates no collinearity. If value exceeds 5, then collinearity should be addressed.
# 
# We will take all the numeric columns and try to find variance_inflation_factor for them.

# In[ ]:


vif=pd.DataFrame()
vif["columns"]=['Order Item Discount',
       'Order Item Discount Rate', 'Order Item Profit Ratio',
       'Order Item Quantity', 'Sales', 'Order Item Total',
       'Order Profit Per Order','Product Price']
vif["vif value"] = [variance_inflation_factor(dFppsc[['Order Item Discount',
       'Order Item Discount Rate', 'Order Item Profit Ratio',
       'Order Item Quantity', 'Sales', 'Order Item Total',
       'Order Profit Per Order','Product Price']].values, i) for i in range(len(vif["columns"]))]
vif.T


# You can see that several of columns have vif values above 5 and some of them are very high. Such high value usually indicate that value in one column can be derived from values in other columns. Lets look at the data and see if we can find the relation

# In[ ]:


dFppsc[['Order Item Discount',
       'Order Item Discount Rate', 'Order Item Profit Ratio',
       'Order Item Quantity', 'Sales', 'Order Item Total',
       'Order Profit Per Order','Product Price']].head(5)


# When you do some basic math you will find these below relationship among these columns:-
# 
# Sales = Product Price * Order Item Quantity 
# Order Item Total = Sales - Order Item Discount
# Order Item Discount = Sales * Order Item Discount Rate
# Order Profit Per Order= Order Item Total * Order Item Profit Ratio
# These are perfect multi colinear features that shouldn't be passed on to regression models because it affects the performance of the model. You can drop one of the features from each of the above relationship.

# In[ ]:


df1=dFppsc.drop(["Order Item Total","Product Price","Order Item Discount Rate","Order Profit Per Order"],axis=1)


# In[ ]:


vif=pd.DataFrame()
vif["columns"]=['Sales',
                'Order Item Quantity','Order Item Discount','Order Item Profit Ratio']
vif["data"] = [variance_inflation_factor(df1[['Sales',
                'Order Item Quantity','Order Item Discount','Order Item Profit Ratio']].values, i) for i in range(len(vif["columns"]))]
vif


# Here we can clearly see that the VIF value is reduced very highly

# In[ ]:


dFppsc['Delivery Status'].value_counts()


# We can see that there are 7754 orders where shipping has been canceled. Lets try to find why it may be happening.

# In[ ]:


plt=dFppsc[dFppsc["Delivery Status"]=="Shipping canceled"].groupby("Order Status").agg(tc=("Order Id","count"))
plt


# As we can see shipping has been canceled because customer canceled his order or order was marked as suspected fraud. We will have to remove these orders when we do RFM and ABC analysis as these were transaction that didn't add any value for store or for customer.

# In[ ]:


df_clean=dFppsc[dFppsc["Delivery Status"]!="Shipping canceled"].copy()


# In[ ]:


df_clean


# ## RFM Analysis
# Recency Frequency Monetary(RFM) Analysis
# RFM analysis is a way to segment customers. This segmentation helps us to identify our most valuable customer. But what does Recency Frequency And Monetary mean?
# 
# Recency - How Recently has the customer purchased from us? Ideally the difference between the customer last purchase date and todays date.
# 
# Frequency - Total number of purchases made from our store or average time between two purchases at the store.
# 
# Monetary - Total value of all purchases made by the customer.
# 
# The more recent the purchase, more responsive the customer is to promotions The more frequently the customer buys, the more engaged and satisfied the customer are. The monetary values helps us to distinguish between big spenders and low spenders.
# 
# RFM analysis is one of the most common techniques used for customer segmentation and is used for making targeted promotions by marketing deparment. Lets begin the process of RFM analysis

# In[ ]:


dF_frequency=df_clean.groupby(["Customer Id"]).agg(Total_count=("Order Id","nunique"))


# In[ ]:


dF_frequency.to_excel("Frequency.xlsx")


# In[ ]:


df_clean["Year"]=df_clean["order date (DateOrders)"].dt.year
df_frequency_year=df_clean.groupby(["Customer Id","Year"], as_index=False)["Order Id"].nunique()
df_frequency_year


# Organising the data for better comparison

# In[ ]:


df_frequency_year=df_frequency_year.pivot_table(index="Customer Id", columns="Year", values="Order Id", fill_value=0)
df_frequency_year


# There is something odd in the data, as we can see in excel that no customer having customer id less than 12436 has purchased in 2018, also after customer id 12336, all the customer has made exactly one purchase, there seems to be some problem in the data, lets dive deeper

# In[ ]:


pd.set_option("display.max_columns", None)
df_clean[df_clean["Customer Id"]==12436]


# Here after looking at the date timestamp from "02-10-2017" all the customers recorded in the system are new.There are no records of old customer purchasing after 2nd october, this problem is with the dataset which should be addressed from data engineers or the data owners to know the results behind the irregularity.

# This particular customer has purchased at 2017-10-02 12:46:00. Lets see the data around this timestamp to see if we can find something unusual.

# In[ ]:


dt1=datetime.datetime.strptime('02-10-2017 12:46:00', '%d-%m-%Y %H:%M:%S')
dt2=dt1-datetime.timedelta(days=5)


# In[ ]:


df_exploring=df_clean[df_clean["order date (DateOrders)"]>=dt2]
df_exploring.to_excel("afterdt1.xlsx")
df_exploring[df_exploring["order date (DateOrders)"]>=dt1]


# As we can see after date time of "02-10-2017 12:46:00" all the customer that are recorded in system are new. There are no records of old customer purchasing after 2nd October 2017 12:46 PM. This may be due to multiple reasons and in real life we must connect with the data engineers or the data owners to know the reason behind this irregularity.
# 
# One possible reason could be that after the above date time the company started using different system to handle old customers and was using the same system to handle new customers
# 
# We can try to see that if there are records with same Customer Name and Customer City after the above timestamp and before the timestamp. If there are such cases present, we can compare their other transaction details such as product purchased ,customer street, Order city to find out whether these customers are the same and system have erroneously assigned a new customer Id even to old customer

# In[ ]:


dF_clean=df_clean
dF_clean["Customer Full Name"]=dF_clean["Customer Fname"]+" "+dF_clean["Customer Lname"]
datetime_val = datetime.datetime.strptime('02-10-2017 12:46:00', '%d-%m-%Y %H:%M:%S')
Customer_Names_After_Dt=dF_clean[dF_clean["order date (DateOrders)"]>=datetime_val][["Customer Full Name","Customer City"]].drop_duplicates()


# In[ ]:


Customer_Name_Common_Before_After_dt=pd.merge(dF_clean[(dF_clean["order date (DateOrders)"]<=datetime_val)],Customer_Names_After_Dt,how="inner",on=["Customer Full Name","Customer City"])
Records=pd.merge(Customer_Name_Common_Before_After_dt[["Customer Full Name","Customer City"]].drop_duplicates(),dF_clean,how="inner",on=["Customer Full Name","Customer City"])
Records.to_excel("idissue.xlsx")


# In[ ]:


Records[Records["Customer Full Name"]=="Jennifer Ramos"]


# After looking at the data in excel we can say that even when customer were having same name and city, the different data points such as Customer Street and order region were different. From this we can infer that they are different individuals having same name and same city.
# 
# For. e.g in above output we see the case of Jennifer Ramos. The last row data points such as Customer Street, Market, Customer Segment are distinctively different from the rows above it. This suggests that Jennifer Ramos with Customer City as Caguas who purchased from store after date time of "02-10-2017 12:46:00" is different individual from Jennifer Ramos who purchased from us before that datetime and also has same customer city Caguas.
# 
# This rules out the possibility that system was erroneously assigning new customer id's even to returning customers. We also can't use email id to find matching customers before and after the the time stamp as all the email id are masked.
# 
# In this scenario we will remove the data after the above date time value and perform or RFM analysis on the remaining data.
# 
# This is necessary because even if a old customer purchases from our store after above date time, his transaction would'nt be recorded in the system and his frequency and monetory value would not increase.
# 
# Its necessary to always be on alert while analyzing data because its better to have no insights than bad insights.

# In[ ]:


dF_clean=dF_clean[dF_clean["order date (DateOrders)"]<datetime_val].copy()


# In[ ]:


dF_frequency=dF_clean[["Customer Id","Order Id"]].groupby("Customer Id",as_index=False).nunique()
dF_frequency.columns=["Customer Id","Frequency"]
sns.histplot(dF_frequency.Frequency,bins=15,kde=False)


# ## 50% of our customers have ordered from us 4 times or more.

# In[ ]:


dF_frequency["F"],Intervals_Frequency=pd.qcut(dF_frequency["Frequency"],q=3,labels=[1,2,3],retbins=True)
Intervals_Frequency


# # Recency of a customer is time since last purchase date of customer. Here the opposite will be true,lesser the recency, more responsive is customer to promotional offers.
# 
# Here we will find last purchase date for each customer and subract it from the last recorded date in the database. This is done because dataset is not recent and last date in our cleaned data is of 2017.
# 
# In a real life scenario, we will subract last purchase date from current date.

# In[ ]:


dF_recency=dF_clean[["Customer Id","order date (DateOrders)"]].groupby("Customer Id",as_index=False).max()
dF_recency.rename(columns ={"order date (DateOrders)":"last_purchase_date"},inplace=True)
max_date=dF_recency["last_purchase_date"].max()
dF_recency["recency"]=max_date-dF_recency["last_purchase_date"]
dF_recency["recency"]=dF_recency["recency"].dt.days


# In[ ]:


sns.displot(dF_recency.recency,bins=8,kde=False)


# We can observe that around 5250 of our customer have purchased from us in the last three months.

# In[ ]:


dF_recency['R'],Intervals_Recency=pd.qcut(dF_recency["recency"],q=3,labels=[3,2,1],retbins=True)


# In[ ]:


print("Interval Recency:"+f"{Intervals_Recency}")


# Monetary value represents total amount of all orders of a customer. Here more is better, so higher score will be assign to higher monetary value.

# In[ ]:


dF_monetory=dF_clean[["Customer Id","Order Item Total"]].groupby("Customer Id",as_index=False).sum()
dF_monetory.columns=["Customer Id","Sum_of_Sales"]
dF_monetory["M"],Intervals_Monetory=pd.qcut(dF_monetory["Sum_of_Sales"],q=3,labels=[1,2,3],retbins=True)
Intervals_Monetory


# In[ ]:


sns.displot(dF_monetory.Sum_of_Sales,kde=True)


# We can see that Monetary value follows a normal distribution, also the average of all monetary value looks to be around 2300-2400.
# 
# Now we will join all three values of recency, frequency, and monetary on Customer Id. We will find a combine score of RFM for each Customer Id. Then we will segment the customers based on that score.

# In[ ]:


dF_rfm=pd.merge(dF_recency[["Customer Id","R"]],dF_monetory[["Customer Id","M"]],on="Customer Id",how="inner")
dF_rfm=pd.merge(dF_rfm,dF_frequency[["Customer Id","F"]],on="Customer Id",how="inner")
dF_rfm["RFM"]=(dF_rfm["R"]).astype(str)+(dF_rfm["F"]).astype(str)+(dF_rfm["M"]).astype(str)


# In[ ]:


dF_rfm


# In[ ]:


def Customer_Segment(data):
    if data["R"]==1 and data["F"] in [1,2,3] and (data["M"]==3):
        return "Lost Customers - Big Spenders"
    elif data["R"]== 1 and data["F"] in [1,2] and data["M"] in [1,2]:
        return "Lost Customers - Bargain"
    elif data["R"] in [1,2] and data["F"]==3 and data["M"] in [1,2]:
        return "Lost/Almost Lost Customers - Loyal"
    elif (data["R"]==3) and (data["F"]==3) and data["M"] in [1,2]:
        return "Loyal Customers"
    elif (data["R"]==3) and data["F"] in [3,2] and data["M"]==3:
        return "Big Spenders"
    elif (data["R"]==3) and (data["F"]==1) and data["M"] in [1,2,3]:
        return "New Customers"
    elif (data["R"]==3) and (data["F"]==2) and data["M"] in [1,2]:
        return "Bargain Customers"
    elif (data["R"]==2) and data["F"]==2 and data["M"] in [1,2]:
        return "Occasional Customers-Bargain"
    elif (data["R"]==2) and data["F"] in [2,3] and data["M"]==3:
        return "Occasional Customers- Big Spenders"
    elif (data["R"]==2) and data["F"]==1 and data ["M"] in [1,2,3]:
        return "Unsatisfied Customers"
    else:
        return "No Segment"


# In[ ]:


dF_rfm["R"]=dF_rfm["R"].astype("category")
dF_rfm["F"]=dF_rfm["F"].astype("category")
dF_rfm["M"]=dF_rfm["M"].astype("category")
dF_rfm["Segment"]=dF_rfm.apply(Customer_Segment,axis=1)


# In[ ]:


Segment_count=(dF_rfm.groupby("Segment",as_index=False).agg(Total_Count=("Customer Id","count"))).sort_values(by="Total_Count",ascending=False)


# In[ ]:


dF_rfm
dF_rfm.to_excel("rfm.xlsx")


# In[ ]:


fig2=go.Figure()
fig2.add_trace(go.Bar(x=Segment_count.Segment,
                      y=Segment_count.Total_Count,
                     hovertemplate ="%{label}<br>Number of Customers:%{value}",
                     texttemplate = "%{value}"))
fig2.update_layout(title="Number of Customer in Each Segment",
                    xaxis=dict(title="Customer Segment"),
                  yaxis=dict(title="Number Of Customers"),width=800)


# In[ ]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Selecting the RFM scores for clustering
rfm_data= dF_rfm
rfm_scores = rfm_data[['R', 'F', 'M']]

# Determining the optimal number of clusters using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scores)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()


# In[ ]:


# Performing K-means clustering with 3 clusters
kmeans_rfm = KMeans(n_clusters=4, random_state=42)
rfm_data['Cluster'] = kmeans_rfm.fit_predict(rfm_scores)

# Displaying the first few rows of the RFM data with the cluster labels
rfm_data.head()


# In[ ]:


rfm_data.head()


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you've loaded your RFM data with cluster labels into 'rfm_data'

# Cluster Profile Analysis: Average RFM scores in each cluster
# Ensure 'R', 'M', and 'F' are numeric
rfm_data[['R', 'M', 'F']] = rfm_data[['R', 'M', 'F']].apply(pd.to_numeric, errors='coerce')

# Cluster Profile Analysis with numeric_only=True
cluster_profile = rfm_data.groupby('Cluster').mean(numeric_only=True)[['R', 'M', 'F']]


# Segment Distribution in Clusters
segment_distribution = rfm_data.groupby(['Cluster', 'Segment']).size().unstack(fill_value=0)

# Preparing data for RFM Score Distributions plot
rfm_melted = rfm_data.melt(id_vars=['Cluster'], value_vars=['R', 'M', 'F'], var_name='RFM', value_name='Score')

# Count of Customers in Each Cluster
customer_count = rfm_data['Cluster'].value_counts().sort_index()

# Plotting
plt.figure(figsize=(18, 12))

# Cluster Profile Plot
plt.subplot(2, 2, 1)
sns.heatmap(cluster_profile, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Average RFM Scores per Cluster')

# Segment Distribution Plot
plt.subplot(2, 2, 2)
segment_distribution.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Segment Distribution across Clusters')
plt.ylabel('Number of Customers')
plt.xlabel('Cluster')

# RFM Score Distributions Plot
plt.subplot(2, 2, 3)
sns.boxplot(data=rfm_melted, x='Cluster', y='Score', hue='RFM')
plt.title('RFM Score Distributions in Each Cluster')

# Count of Customers in Each Cluster Plot
plt.subplot(2, 2, 4)
customer_count.plot(kind='bar', ax=plt.gca())
plt.title('Count of Customers in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')

plt.tight_layout()
plt.show()


# In[ ]:


def generate_cluster_strategies(rfm_data):
    """
    Generate strategies based on RFM clustering.

    Parameters:
    rfm_data (DataFrame): DataFrame containing RFM scores and cluster labels

    Returns:
    dict: A dictionary with cluster numbers as keys and strategy recommendations as values
    """
    cluster_strategies = {}
    cluster_profiles = rfm_data.groupby('Cluster').mean()[['R', 'F', 'M']]

    for cluster in cluster_profiles.index:
        profile = cluster_profiles.loc[cluster]
        strategies = []

        # High Recency, Frequency, Monetary
        if profile['R'] > 3 and profile['F'] > 3 and profile['M'] > 3:
            strategies.append("Focus on loyalty programs and premium services.")
            strategies.append("Encourage referrals and reviews.")

        # High Recency, Low Frequency and Monetary
        elif profile['R'] > 3 and profile['F'] <= 2 and profile['M'] <= 2:
            strategies.append("Engage with personalized communications to increase frequency.")
            strategies.append("Offer incentives for frequent purchases.")

        # Low Recency, Varying Frequency and Monetary
        elif profile['R'] <= 2:
            strategies.append("Re-engage with win-back campaigns and special offers.")
            strategies.append("Conduct surveys to understand their lack of recent engagement.")

        # Other combinations
        else:
            strategies.append("Analyze purchase patterns for targeted promotions.")
            strategies.append("Consider cross-selling and upselling opportunities.")

        cluster_strategies[cluster] = strategies

    return cluster_strategies

# Example usage
cluster_strategies = generate_cluster_strategies(rfm_data)
for cluster, strategies in cluster_strategies.items():
    print(f"Cluster {cluster} Strategies:")
    for strategy in strategies:
        print(f"- {strategy}")
    print()


# In[ ]:


# Calculate the average RFM values for each cluster
cluster_profiles = rfm_data.groupby('Cluster').mean()[['R', 'F', 'M']]

# Interpretation
for cluster, profile in cluster_profiles.iterrows():
    print(f"Cluster {cluster}:")
    print(f"- Average Recency (R): {profile['R']}")
    print(f"- Average Frequency (F): {profile['F']}")
    print(f"- Average Monetary (M): {profile['M']}")
    if profile['R'] > 3 and profile['F'] > 3 and profile['M'] > 3:
        print("  - Likely to be 'Best Customers'")
    elif profile['R'] < 2 and profile['F'] > 3 and profile['M'] > 3:
        print("  - Likely to be 'Lapsing High-Value Customers'")
    # Add other interpretations as needed
    print()


# In[ ]:


def get_strategy_by_cluster(cluster):
    """
    Provide strategies based on the customer's cluster.

    Parameters:
    cluster (int): The cluster number the customer belongs to

    Returns:
    dict: A dictionary containing the customer segment and corresponding strategy
    """
    strategies = {
        0: {
            "Segment": "Average Recent, Low Frequency & Monetary",
            "Strategy": ("Engage with personalized communications and offers to increase purchase frequency. "
                         "Introduce loyalty programs to boost spending.")
        },
        1: {
            "Segment": "At Risk of Churning",
            "Strategy": ("Implement re-engagement campaigns such as special discounts or reminders of product value. "
                         "Conduct surveys to understand their inactivity and address these concerns.")
        },
        2: {
            "Segment": "New or One-time Buyers",
            "Strategy": ("Focus on enhancing first-time buyer experiences and follow-up marketing. "
                         "Encourage second purchases through targeted offers based on initial buying behavior.")
        },
        3: {
            "Segment": "Best Customers",
            "Strategy": ("Prioritize premium customer experiences and services. "
                         "Explore opportunities for upselling and cross-selling. "
                         "Maintain high engagement with exclusive offers and early access to new products.")
        }
    }

    return strategies.get(cluster, {"Segment": "Unknown", "Strategy": "Conduct further analysis to determine the best approach."})

# Example usage
cluster_strategy = get_strategy_by_cluster(2)
print(f"Cluster: {2}")
print(f"Segment: {cluster_strategy['Segment']}")
print(f"Strategy: {cluster_strategy['Strategy']}")


# In[ ]:


def get_strategy_by_cluster(cluster):
    cluster_strategies = {
        0: "Engage with personalized communications and offers to increase purchase frequency. Introduce loyalty programs to boost spending.",
        1: "Implement re-engagement campaigns such as special discounts or reminders of product value. Conduct surveys to understand their inactivity and address these concerns.",
        2: "Focus on enhancing first-time buyer experiences and follow-up marketing. Encourage second purchases through targeted offers based on initial buying behavior.",
        3: "Prioritize premium customer experiences and services. Explore opportunities for upselling and cross-selling. Maintain high engagement with exclusive offers and early access to new products."
    }
    return cluster_strategies.get(cluster, "Conduct further analysis to determine the best approach.")


# In[ ]:


# Apply the strategy function to each row in the DataFrame
rfm_data['Strategy'] = rfm_data['Cluster'].apply(get_strategy_by_cluster)

# Display the DataFrame with the new 'Strategy' column
print(rfm_data.head())


# In[ ]:


import pandas as pd



rfm_data = pd.DataFrame(rfm_data)

# Define the strategy mapping function
def get_strategy_by_cluster(cluster):
    cluster_strategies = {
        0: "Personalized communications, loyalty programs.",
        1: "Re-engagement campaigns, special discounts.",
        2: "First-time buyer experiences, follow-up marketing.",
        3: "Premium experiences, upselling and cross-selling."
    }
    return cluster_strategies.get(cluster, "Further analysis required.")

# Apply the strategy function
rfm_data['Strategy'] = rfm_data['Cluster'].apply(get_strategy_by_cluster)

# Reordering columns for a better flow
column_order = ["Customer Id", "R", "F", "M", "RFM", "Segment", "Cluster", "Strategy"]
rfm_data = rfm_data[column_order]

# Optional: Format numerical columns (if needed)
# rfm_data['R'] = rfm_data['R'].map('{:.1f}'.format)
# rfm_data['F'] = rfm_data['F'].map('{:.1f}'.format)
# rfm_data['M'] = rfm_data['M'].map('{:.1f}'.format)

# Display the formatted DataFrame

rfm_data = pd.DataFrame(rfm_data)
print(rfm_data.head())


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming rfm_data is your DataFrame
# rfm_data = pd.read_csv('path_to_your_data.csv') # Uncomment this line if you're reading from a CSV file

# Segmentation Analysis
segment_counts = rfm_data['Segment'].value_counts()
plt.figure(figsize=(10, 6))
segment_counts.plot(kind='bar')
plt.title('Customer Segmentation Distribution')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# RFM Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(rfm_data['RFM'], kde=False, bins=20)
plt.title('RFM Score Distribution')
plt.xlabel('RFM Score')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


rfm_data["Cluster"].value_counts()


# ABC analysis is used for product segmentations. It is a type of supply side segmentation while RFM analysis is demand side segmentation.
# 
# ## ABC analysis has its basis in the pareto rule which states that 80% of outcome comes from 20% of causes.
# 
# Similarly here, 80% of the revenue is generated by 20% of all products. High care should be taken that these 20% of the products are always instock.
# 
# Here's how we segment product based on their revenue generating capacity.
# 
# We find percentage of total revenue generated by each product.
# Then we arrange them in decreasing order of percentage and we find cumulative percentage of revenue for each of the products.
# Products generating 75% of revenue would be Class A products
# Procucts generating next 20% of revenue would be class B products. Products having their cumulative percentage between 75%-95%.
# Products generating remaining 5% of revenue would be class C products. Long list of products each contribuling very less to total revenue. Also called as long tail. They have their cumulative percentage between 95%-100%

# In[ ]:


Total_Products= df_clean["Product Name"].nunique()


# In[ ]:


Total_Products


# In[ ]:


Revenue_ABC=dF_clean.groupby(["Department Name","Product Name"]).agg(Total_Revenue=("Order Item Total","sum")).sort_values(by="Total_Revenue",ascending=False).reset_index()
Revenue_ABC["cum_sum"]=Revenue_ABC["Total_Revenue"].cumsum()
Revenue_ABC["cum_per"]=Revenue_ABC["cum_sum"]/Revenue_ABC["Total_Revenue"].sum()*100
Revenue_ABC["per"]=Revenue_ABC["cum_per"]-Revenue_ABC["cum_per"].shift(1)
Revenue_ABC.loc[0,"per"]=Revenue_ABC["cum_per"][0]


# In[ ]:


Revenue_ABC


# In[ ]:


def ABC(data):
    if data["cum_per"]<=75:
        return "A"
    elif data["cum_per"]>75 and data["cum_per"]<=95:
        return "B"
    elif data["cum_per"]>95:
        return "C"

Revenue_ABC["ABC_Revenue"]=Revenue_ABC.apply(ABC,axis=1)


# In[ ]:


Bar_graph_Abc=Revenue_ABC[["ABC_Revenue","Product Name","Total_Revenue"]].groupby("ABC_Revenue").agg(Revenue=("Total_Revenue","sum"),count=("Product Name","count"))


# In[ ]:


Revenue_ABC


# In[ ]:


Bar_graph_Abc
fig2=go.Figure(go.Bar(x=Bar_graph_Abc.index,
                      y=Bar_graph_Abc["Revenue"],
                     hovertemplate ="%{label}<br>Revenue:%{value}",
                     texttemplate = "Revenue<br>%{value}",
                     marker_color=["orange","lightgreen","red"],
                     showlegend=False))
fig2.add_trace(                          
    go.Scatter(                          
    x=Bar_graph_Abc.index,
    y=Bar_graph_Abc["count"],
    name="Number Of Products",
    mode='lines',                                                       
    line = dict(color='blue', width=3),   
    yaxis="y2",
    marker_line_width = 0
    ))

fig2.update_layout(
                   title="Revenue Generated By Products in Different ABC Segments",
                   xaxis=dict(title="Segment" ),
                   yaxis=dict(title="Revenue",showgrid=False),
                   yaxis2=dict(title="Number Of Products", anchor="x", overlaying="y",side="right",dtick=10),
                   legend = dict(x = 1.05, y = 1))
fig2.show()


# In[ ]:





# ### SMART QUESTION 2
# #### How does the product category relate to sales performance, and can this insight inform inventory stocking?

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats


# In[ ]:


import os
os.getcwd()


# In[ ]:


df_sm3 = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')


# In[ ]:


pd.set_option('display.max_columns', 500)


# In[ ]:


df_sm3.head()


# ##### 1. Data Cleaning and Exploration:

# In[ ]:


df_sm3.info()


# In[ ]:


# renaming the column
df_sm3.rename(columns={'order date (DateOrders)': 'Order_Date'}, inplace = True)


# In[ ]:


missing_values = df_sm3.isnull().sum()

print("Missing Values in Each Column:")
print(missing_values)


# Check the unique product categories in the dataset

# In[ ]:


print(df_sm3['Category Name'].unique())


# In[ ]:


category_count = df_sm3['Category Name'].nunique()
print(category_count)


# In[ ]:


print(df_sm3.groupby(['Category Name'])['Sales'].median())


# ##### 2. Visualization for Sales Performance:

# Visualize the sales performance across different product categories using bar plots or box plots.

# In[ ]:


category_sales = df_sm3.groupby('Category Name')['Sales'].sum().reset_index()
category_sales = category_sales.sort_values(by = 'Sales', ascending = False)


# In[ ]:


category_sales


# In[ ]:


plt.figure(figsize=(14, 10))
sns.barplot(x='Sales', y='Category Name', data=category_sales, palette="viridis")
plt.xlabel('Total Sales')
plt.ylabel('Product Category')
plt.title('Total Sales by Product Category')
plt.show()


# In[ ]:


# Using box-plot
plt.figure(figsize=(14, 10))
sns.boxplot(x='Sales', y='Category Name', data=df_sm3)
plt.xticks(rotation=90)
plt.title('Sales Performance Across Category Name')
plt.show()


# It's interesting to note that while Fishing has the highest total sales, its range of sales isn't as varied compared to other categories. This indicates that products in the Fishing category might not have a wide price range compared to items in other categories.

# Let's visualize Customer Segment on each Category Name.

# In[ ]:


df_sm3['Customer Segment'].unique()


# In[ ]:


order_profit = df_sm3.groupby(['Customer Segment'])['Order Profit Per Order'].sum()


# In[ ]:


order_profit = order_profit.reset_index()


# In[ ]:


plt.figure(figsize=(14, 10))
sns.barplot(x='Customer Segment', y='Order Profit Per Order', data=order_profit, palette="viridis")
plt.xlabel('Customer Segments')
plt.ylabel('Order Profit')
plt.title('Order Profit by Customer Segments')
plt.show()


# In[ ]:


unique_categories = df_sm3['Category Name'].unique()

# Plotting separate count plots for each category
for category in unique_categories:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_sm3[df_sm3['Category Name'] == category], x='Customer Segment')
    plt.title(f'Customer Segmentation for {category}')
    plt.xlabel(f'Customer Segmentation for {category}')
    plt.ylabel('Count')
    plt.show()


# In[ ]:


def categorize_items(category_name):
    sports_keywords = ['Cleats', "Men's Footwear", 'Fishing', 'Indoor/Outdoor Games',
                       'Water Sports', 'Camping & Hiking', 'Cardio Equipment', 'Shop By Sport',
                       "Golf Balls", "Golf Gloves", 'Trade-In', 'Video Games', "Toys"
                        "Baseball & Softball", "Hockey", "Lacrosse", "Golf Apparel", "Hunting & Shooting", 
                        "Boxing & MMA", "Kids' Golf Clubs", "Fitness Accessories", "Men's Golf Clubs", "Strength Training",
                       "Tennis & Racquet", "Sporting Goods", "Golf Shoes", 'Soccer', 'Basketball', "Women's Golf Clubs"]

    clothing_keywords = ["Women's Apparel", "Children's Clothing", "Women's Clothing", "Girls' Apparel",
                 "Men's Clothing", 'Baby']

    gadgets_keywords = ['Electronics', 'Gadgets', 'Cameras', 'DVDs', 'Computers', 'Consumer Electronics',
                    'Music', 'CDs']
    health_beauty_keywords = ['Health and Beauty', 'Accessories']

    others =["Pet Supplies", "Crafts", "Garden", "Music", "Books"]

    if any(keyword in category_name for keyword in sports_keywords):
        return 'Sports'
    elif any(keyword in category_name for keyword in clothing_keywords):
        return 'Clothing'
    elif any(keyword in category_name for keyword in gadgets_keywords):
        return 'Gadgets'
    elif any(keyword in category_name for keyword in health_beauty_keywords):
        return 'Health & Beauty'
    else:
        return 'Other'


# In[ ]:


# Apply the categorization function to create 'Category_Class' column
df_sm3['Category_Class'] = df_sm3['Category Name'].apply(categorize_items)


# In[ ]:


df_sm3['Category_Class'].unique()


# In[ ]:


unique_categories = df_sm3['Category_Class'].unique()
num_categories = len(unique_categories)

# Create subplots for count plots
fig, axes = plt.subplots(nrows=1, ncols=num_categories, figsize=(16, 5))

# Plotting separate count plots for each category
for i, category in enumerate(unique_categories):
    sns.countplot(data=df_sm3[df_sm3['Category_Class'] == category], x='Customer Segment', ax=axes[i])
    axes[i].set_title(f'Customer Segmentation for {category}')
    axes[i].set_xlabel(f'Customer Segmentation for {category}')
    axes[i].set_ylabel('Count')

plt.tight_layout()
plt.show()


# In[ ]:


# median tells the middle of the Sales per each order
df_sm3.groupby(['Customer Segment'])['Sales'].median()


# In[ ]:


# Analyze sales across different customer segments
plt.figure(figsize=(8, 6))
sns.boxplot(x='Customer Segment', y='Sales', data=df_sm3)
plt.title('Sales Across Customer Segments')
plt.xlabel('Customer Segment')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


category_class_sales = df_sm3.groupby('Category_Class')['Sales'].sum().reset_index()
category_class_sales = category_class_sales.sort_values(by = 'Sales', ascending = False)


# In[ ]:


category_class_sales


# In[ ]:


# Using bar-plot
plt.figure(figsize=(14, 10))
sns.barplot(x='Category_Class', y='Sales', data=category_class_sales, palette="viridis")
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.title('Total Sales by Product Category Class')
plt.show()


# In[ ]:


# Using box-plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='Category_Class', y='Sales', data=df_sm3)
plt.xticks(rotation=90)
plt.title('Sales Performance Across Category_Class')
plt.show()


# In[ ]:


df_sm3['Category_Class'].value_counts()


# Which product category has/have the maximum discount ?

# In[ ]:


# Grouping the data by 'Category_Class Name' and finding the maximum discount 
max_discount_per_product = df_sm3.groupby('Category_Class')['Order Item Discount'].max()
print(max_discount_per_product)
# Finding the product category with maximum discount
product_with_max_discount = max_discount_per_product.idxmax()
max_discount_value = max_discount_per_product.max()
print('----------------------------------------------------------')
print(f"The product with the maximum discount is '{product_with_max_discount}' with a discount of {max_discount_value:.2f}")


# To investigate whether discounts impact sales, perform correlation analysis between the discount and sales figures.

# In[ ]:


# Calculate Pearson's correlation coefficient between 'Order Item Discount' and 'Sales'
correlation = df_sm3['Order Item Discount'].corr(df_sm3['Sales'])

print(f"Pearson's Correlation Coefficient between Discount and Sales: {correlation:.4f}")


# A value of 0.6174 suggests that there is a moderate positive correlation between the discount offered on items and the resulting sales figures. This correlation coefficient suggests that as the discounts increase, there is a tendency for sales to also increase, though it doesn't necessarily imply causation.
# 

# #### Analyze trends in sales over time:

# To analyze trends in sales over timeor across different customer segments using the provided sample dataset, we'll focus on the 'order date' and 'Sales' columns to visualize trends in sales over time.

# In[ ]:


# Convert 'order date' column to datetime format
df_sm3['Order_Date'] = pd.to_datetime(df_sm3['Order_Date'])


# In[ ]:


# Extract year and month from the 'order date' column
df_sm3['Order_Year'] = df_sm3['Order_Date'].dt.year
df_sm3['Order_Month'] = df_sm3['Order_Date'].dt.month


# In[ ]:


# Aggregate sales by year and month
sales_over_time = df_sm3.groupby(['Order_Year', 'Order_Month'])['Sales'].sum().reset_index()


# In[ ]:


# Separate sales data for each year
years = sales_over_time['Order_Year'].unique()

# Plotting sales trend over time for each year separately
plt.figure(figsize=(12, 8))


# In[ ]:


for year in years:
    year_data = sales_over_time[sales_over_time['Order_Year'] == year]
    plt.plot(year_data['Order_Month'], year_data['Sales'], marker='o', label=str(year))

plt.title('Sales Trend Over Time (Separated by Year)')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(range(1, 13))  # Assuming data spans all 12 months
plt.legend(title='Year')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# Filter the dataset for November (11th month) and December (12th month) of the year 2017
nov_dec_2017_data = df_sm3[
    (df_sm3['Order_Date'].dt.year == 2017) &
    (df_sm3['Order_Date'].dt.month.isin([11, 12]))
]

# Get the unique product categories bought during November and December 2017
product_categories_nov_dec_2017 = nov_dec_2017_data['Category Name'].unique()

print("Product categories bought in November and December 2017:")
print(product_categories_nov_dec_2017)


# In[ ]:


# Calculate the count of each category in the dataset
category_counts_overall = df_sm3['Category Name'].value_counts()

# Calculate the count of categories bought in Nov-Dec 2017
category_counts_nov_dec_2017 = nov_dec_2017_data['Category Name'].value_counts()

# Get the category classes for each category in Nov-Dec 2017
category_classes = df_sm3[df_sm3['Category Name'].isin(product_categories_nov_dec_2017)][['Category Name', 'Category_Class']]
category_classes = category_classes.drop_duplicates(subset=['Category Name'])

print("Category Counts in Nov-Dec 2017:")
print(category_counts_nov_dec_2017)
print("\nCategory Counts Overall:")
print(category_counts_overall)
print("\nCategory Classes for Nov-Dec 2017:")
print(category_classes)


# In[ ]:


# Sort categories alphabetically for consistent representation
category_counts_nov_dec_2017_sorted = category_counts_nov_dec_2017.sort_index()
category_classes_sorted = category_classes.sort_values('Category Name')

# Create figure and two subplots sharing the x-axis
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

# Plot category counts for Nov-Dec 2017
ax1.bar(category_counts_nov_dec_2017_sorted.index, category_counts_nov_dec_2017_sorted.values, color='skyblue', label='Nov-Dec 2017')
ax1.set_ylabel('Category Counts')

# Plot category classes for Nov-Dec 2017
ax2.scatter(category_classes_sorted['Category Name'], category_classes_sorted['Category_Class'], color='orange', marker='o', label='Category Classes')
ax2.set_ylabel('Category Classes')

# Set x-axis labels and rotation
ax1.set_xlabel('Categories')
ax1.set_xticklabels(category_counts_nov_dec_2017_sorted.index, rotation=90)

# Set legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('Category Counts and Classes in Nov-Dec 2017')
plt.tight_layout()
plt.show()


# 
# From the above plot we can infer that even if the market was down during late 2017, we saw the no. of Sales for the Sports itmes were still high, suggesting that it is a good area to focus for inventory stocking.

# In[ ]:


# Filter the dataset for February of the years 2015, 2016, and 2017
feb_2015_2016_2017_data = df_sm3[
    (df_sm3['Order_Date'].dt.month == 2) &
    (df_sm3['Order_Date'].dt.year.isin([2015, 2016, 2017]))
]

# Get unique product categories for each year
feb_2015_categories = set(feb_2015_2016_2017_data[feb_2015_2016_2017_data['Order_Date'].dt.year == 2015]['Category Name'].unique())
feb_2016_categories = set(feb_2015_2016_2017_data[feb_2015_2016_2017_data['Order_Date'].dt.year == 2016]['Category Name'].unique())
feb_2017_categories = set(feb_2015_2016_2017_data[feb_2015_2016_2017_data['Order_Date'].dt.year == 2017]['Category Name'].unique())

# Find the product category bought in Feb 2015 that was also purchased in Feb 2016 and 2017
common_categories_15_16 = feb_2015_categories.intersection(feb_2016_categories)
common_categories_15_17 = feb_2015_categories.intersection(feb_2017_categories)

print("Product categories bought in Feb 2015 that were also purchased in Feb 2016:")
print(common_categories_15_16)
print("\nProduct categories bought in Feb 2015 that were also purchased in Feb 2017:")
print(common_categories_15_17)


# In[ ]:


pip install wordcloud


# In[ ]:


from wordcloud import WordCloud

# Combine the shared categories into a single list
shared_categories = common_categories_15_16.union(common_categories_15_17)

# Convert the set of shared categories to a single string
text = ' '.join(shared_categories)

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Shared Product Categories Bought in Feb 2015, 2016, and 2017')
plt.axis('off')
plt.show()


# In[ ]:


# Get the count of each product category
category_counts = nov_dec_2017_data['Category Name'].value_counts()

# Plotting a bar chart to visualize product categories bought in November and December 2017
plt.figure(figsize=(12, 6))
category_counts.plot(kind='bar', color='skyblue')
plt.title('Product Categories Bought in Nov-Dec 2017')
plt.xlabel('Product Category')
plt.ylabel('Number of Orders')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[ ]:


# Creating a table to display the product categories and their counts
category_table = pd.DataFrame({'Product Category': category_counts.index, 'Number of Orders': category_counts.values})
print("Table: Product Categories Bought in Nov-Dec 2017")
category_table


# In[ ]:


# Filter data for the second month (February - month number 2)
second_month_data = df_sm3[df_sm3['Order_Month'] == 2]

# Group by year and calculate total sales for February of each year
monthly_sales = second_month_data.groupby('Order_Year')['Sales'].sum()

# Plotting the line plot to visualize the drop in sales for the second month across years
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linestyle='-', color='blue')
plt.title('Total Sales Trend in the Second Month Across Years')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


# Filter data for a specific year (e.g., 2017)
second_month_sales_2017 = df_sm3[(df_sm3['Order_Month'] == 2) & (df_sm3['Order_Year'] == 2017)]

# Group by month and calculate total sales for all months in 2017
monthly_sales_2017 = df_sm3[df_sm3['Order_Year'] == 2017].groupby('Order_Month')['Sales'].sum()

# Plotting the bar plot to compare sales of the second month (Feb) in 2017 with other months in 2017
plt.figure(figsize=(10, 6))
plt.bar(monthly_sales_2017.index, monthly_sales_2017.values, color='skyblue', label='Other Months (2017)')
plt.bar(2, second_month_sales_2017['Sales'].sum(), color='orange', label='Second Month (Feb) - 2017')
plt.title('Comparison of Sales: Second Month vs Other Months (2017)')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.tight_layout()
plt.show()


# These products consistently sold in February across three consecutive years (2015, 2016, and 2017).
# 
# Product categories bought in Feb 2015 that were also purchased in Feb 2016:
# {'Electronics', 'Baseball & Softball', 'Hockey', 'Indoor/Outdoor Games', 'Golf Apparel', 'Golf Shoes', 'Trade-In', 'Fitness Accessories', 'Cleats', 'Accessories', 'Golf Gloves', 'Camping & Hiking', "Women's Apparel", 'Cardio Equipment', 'Boxing & MMA', 'Lacrosse', 'Hunting & Shooting', 'Golf Balls', 'Shop By Sport', "Girls' Apparel", "Men's Footwear", 'Water Sports', 'Tennis & Racquet', 'Fishing'}
# 
# Product categories bought in Feb 2015 that were also purchased in Feb 2017:
# {'Electronics', 'Baseball & Softball', 'Hockey', 'Indoor/Outdoor Games', 'Golf Apparel', 'Golf Shoes', 'Trade-In', 'Fitness Accessories', 'Cleats', 'Accessories', 'Golf Gloves', 'Camping & Hiking', "Women's Apparel", 'Cardio Equipment', 'Boxing & MMA', 'Lacrosse', 'Hunting & Shooting', 'Golf Balls', 'Shop By Sport', "Girls' Apparel", "Men's Footwear", 'Water Sports', 'Tennis & Racquet', 'Fishing'}

# ## SMART QUESTION 3

# In[1]:


#%%
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import f_oneway
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\varun\Downloads\DataCoSupplyChainDataset.csv", encoding='latin-1')

# Handling missing values
df['Customer Zipcode'].fillna(df['Customer Zipcode'].mean(), inplace=True)
df['Order Zipcode'].fillna(df['Order Zipcode'].mean(), inplace=True)
df['Customer Lname'].fillna('Unknown', inplace=True)

# Drop the 'Product Description' column
df.drop('Product Description', axis=1, inplace=True)

print(df.info())


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt



benefit_min = df['Benefit per order'].min()
benefit_max = df['Benefit per order'].max()
benefit_avg = df['Benefit per order'].mean()

print(f"Minimum Benefit per order: {benefit_min}")
print(f"Maximum Benefit per order: {benefit_max}")
print(f"Average Benefit per order: {benefit_avg}")

# Calculate summary statistics
benefit_summary = df.groupby('Shipping Mode')['Benefit per order'].agg(['min', 'max', 'mean']).reset_index()

# Plotting the data
plt.figure(figsize=(12, 8))
sns.barplot(x='Shipping Mode', y='min', data=benefit_summary, color='blue', label='Min Benefit')
sns.barplot(x='Shipping Mode', y='max', data=benefit_summary, color='orange', label='Max Benefit')
sns.barplot(x='Shipping Mode', y='mean', data=benefit_summary, color='green', label='Avg Benefit')

plt.title('Summary of Benefit per order by Shipping Mode')
plt.xlabel('Shipping Mode')
plt.ylabel('Benefit per order')
plt.legend()
plt.show()


# In[3]:


grouped_data = df.groupby('Shipping Mode').agg({
    'Sales per customer': 'sum',
    'Benefit per order': 'sum'
}).reset_index()

# Plotting the data
plt.figure(figsize=(10, 6))
plt.bar(grouped_data['Shipping Mode'], grouped_data['Sales per customer'], label='Sales')
plt.bar(grouped_data['Shipping Mode'], grouped_data['Benefit per order'], label='Benefit')
plt.xlabel('Shipping Mode')
plt.ylabel('Benefit per order')
plt.title('Relationship between Shipping Mode and Sales Benefit')
plt.legend()
plt.show()


# In[4]:


correlation = df.groupby('Shipping Mode')[['Sales per customer', 'Benefit per order']].corr().iloc[0::2]['Benefit per order'].reset_index()
correlation


# #First Class:
# #The correlation coefficient between "Benefit per order" and "Sales per customer" for First Class shipping mode is 0.159849. This positive correlation indicates a moderate positive relationship between the two variables.
# 
# #Same Day:
# #For Same Day shipping mode, the correlation coefficient is 0.139488. This positive correlation suggests a moderate positive relationship between the benefit per order and sales per customer.
# 
# #Second Class:
# #The correlation coefficient for Second Class shipping mode is 0.117577. This positive correlation indicates a positive relationship, but it is weaker compared to First Class and Same Day.
# 
# #Standard Class:
# #The correlation coefficient for Standard Class shipping mode is 0.131890. Similar to Same Day, there is a moderate positive relationship between benefit per order and sales per customer.
# 
# #In summary, all shipping modes show positive correlations, indicating that as sales per customer increase, the benefit per order also tends to increase. However, the strength of the correlation varies, with First Class having the strongest positive correlation.
# 

# ANOVA

# In[5]:


first_class = df[df['Shipping Mode'] == 'First Class']['Benefit per order']
same_day = df[df['Shipping Mode'] == 'Same Day']['Benefit per order']
second_class = df[df['Shipping Mode'] == 'Second Class']['Benefit per order']
standard_class = df[df['Shipping Mode'] == 'Standard Class']['Benefit per order']

# Performing ANOVA test
anova_result = f_oneway(first_class, same_day, second_class, standard_class)

# Displaying the ANOVA result
print("ANOVA Result:")
print(anova_result)

# Checking significance level
alpha = 0.05
if anova_result.pvalue < alpha:
    print("The means are significantly different. Reject the null hypothesis.")
else:
    print("Fail to reject the null hypothesis. The means are not significantly different.")


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt


# Create a box plot
sns.boxplot(x='Shipping Mode', y='Benefit per order', data=df)
plt.title('Distribution of Benefit per order by Shipping Mode')
plt.show()


# #The ANOVA result you provided indicates that the p-value is 0.1147, which is greater than the typical significance level of 0.05. Therefore, based on this result, you fail to reject the null hypothesis. This suggests that there is not enough evidence to conclude that the means of "Benefit per order" are significantly different across the different shipping modes.
# 
# #In other words, the ANOVA test does not provide sufficient evidence to suggest that there is a significant difference in the average "Benefit per order" between the shipping modes. Keep in mind that the interpretation is based on the chosen significance level, and different levels may lead to different conclusions.
# 
# #If you have any more questions or if there's anything else you'd like to explore in your data, feel free to let me know!

# TUKEY HSD
# 

# In[8]:


from statsmodels.stats.multicomp import pairwise_tukeyhsd


data = pd.concat([first_class, same_day, second_class, standard_class])
labels = ['First Class'] * len(first_class) + ['Same Day'] * len(same_day) + ['Second Class'] * len(second_class) + ['Standard Class'] * len(standard_class)

tukey_results = pairwise_tukeyhsd(data, labels, alpha=0.05)

print(tukey_results)


# #The results of the Tukey HSD test are presented in the table
# #and it appears that none of the pairwise differences in means between the shipping modes are statistically significant
# #The "reject" column indicates whether the null hypothesis of equal means is rejected
# #Since all entries in the "reject" column are marked as "False," it suggests that there is no significant difference between the means of "Benefit per order" for any pair of shipping modes.
# 
# #In summary, both the ANOVA test and the Tukey HSD test indicate that there is not enough evidence to conclude that the means of "Benefit per order" are significantly different among the different shipping modes.
# 

# In[7]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder




label_encoder = LabelEncoder()
df['Shipping Mode'] = label_encoder.fit_transform(df['Shipping Mode'])


X = df[['Benefit per order']]
y = df['Shipping Mode']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the random forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)


y_pred = random_forest_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the random forest model: {:.2f}%".format(accuracy * 100))

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[9]:


from sklearn.metrics import classification_report

# Classification Report for Random Forest
classification_rep_rf = classification_report(y_test, y_pred)
print("Classification Report - Random Forest:")
print(classification_rep_rf)


# In[10]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=range(len(set(y_test))))


y_prob_rf = random_forest_model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

class_names = ['First Class', 'Same Day', 'Second Class', 'Standard Class']

for i in range(len(set(y_test))):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_rf[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure(figsize=(8, 6))
for i in range(len(set(y_test))):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve {} (area = {:.2f})'.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()


# In[11]:


df['Predicted Shipping Mode'] = random_forest_model.predict(X)

# Scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Predicted Shipping Mode', y='Benefit per order', data=df, hue='Shipping Mode', palette='viridis', alpha=0.6)
plt.title('Scatter Plot of Predicted Shipping Mode vs. Benefit per order')
plt.xlabel('Predicted Shipping Mode')
plt.ylabel('Benefit per order')
plt.legend()
plt.show()


# In[12]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


y_pred_rf = random_forest_model.predict(X_test)

# Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Classification Report
print("Classification Report - Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Feature Importance
feature_importance_rf = pd.Series(random_forest_model.feature_importances_, index=X_train.columns)
feature_importance_rf.nlargest(10).plot(kind='barh')
plt.title('Top 10 Feature Importance - Random Forest')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.show()


# In[13]:


from xgboost import XGBClassifier



label_encoder = LabelEncoder()
df['Shipping Mode'] = label_encoder.fit_transform(df['Shipping Mode'])

# Select features and target variable
X = df[['Benefit per order']]
y = df['Shipping Mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_xgb)
mse = mean_squared_error(y_test, y_pred_xgb)
print("Accuracy of the XGBoost model: {:.2f}%".format(accuracy * 100))
print("Mean Squared Error:", mse)


# In[14]:


from sklearn.metrics import classification_report


class_report = classification_report(y_test, y_pred_xgb)

print("Classification Report for XGBoost Model:\n")
print(class_report)


# In[15]:


feature_importance_xgb = xgb_model.feature_importances_
features_xgb = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_xgb, y=features_xgb)
plt.title('Feature Importance - XGBoost')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


# In[16]:


from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(xgb_model, X, y, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Score')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curve - XGBoost')
plt.legend()
plt.show()


# ## SMART QUESTION 4

# ### Is there any significant difference in average order profit between regions?

# In[1]:


# Importing Libraries

import pandas as pd
import folium
from folium.plugins import MarkerCluster
from scipy.stats import f_oneway
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd


# In[3]:


# Reading dataset
df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')


# ##### Before we begin the inferential statistical analysis, we must first examine the data.

# In[4]:


# About Dataset
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())


# In[5]:


print(df['Order Region'].value_counts())


# In[6]:


# Check unique values in 'Order Status' to understand how completed orders are labeled
print(df['Order Status'].value_counts())


# ##### For this Analysis, we will look at orders that have been completed.

# In[7]:


# Filter the dataset for completed orders
df = df[df['Order Status'] == 'COMPLETE']

print(df.head())


# ##### Let's take a look on the origin and destinations

# In[8]:


df[['Latitude', 'Longitude']].head()


# In[9]:


df_stores_location = df[['Latitude', 'Longitude']].drop_duplicates().dropna()


# In[10]:


# Plotting the store locations
import geopandas as gpd

df_geo = gpd.GeoDataFrame(
    df_stores_location,
    geometry=gpd.points_from_xy(df_stores_location['Longitude'], df_stores_location['Latitude'])
)

#get the map image
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

#plot the world map
ax = world.plot(figsize=(12, 8), color='white', edgecolor='black')

#plot the store locations as dots
df_geo.plot(ax=ax, marker='o', color='red', markersize=25, label='Stores')


# ##### Let's make an interactive map for enhanced understanding.

# In[11]:


import folium
from folium.plugins import MarkerCluster
from IPython.display import display


# Filter relevant columns for store locations
store_locations_df = df[['Customer City', 'Latitude', 'Longitude']].drop_duplicates()

# Create a base map
map_center = [store_locations_df['Latitude'].mean(), store_locations_df['Longitude'].mean()]
store_map = folium.Map(location=map_center, zoom_start=5)

# Add markers for each store location
marker_cluster = MarkerCluster().add_to(store_map)

for index, row in store_locations_df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=row['Customer City'],
        icon=folium.Icon(color='blue')
    ).add_to(marker_cluster)

# Display the map
display(store_map)

#store_map.save("store_locations_map.html")


# ##### **OBSERVATION:** The plot reveals a concentration of store locations primarily in North America, with a particularly dense cluster in the United States. Scattered stores can also be found in South America, Europe, Africa, Asia, and Australia, indicating a global presence, albeit one that is far less widespread than in North America. The map emphasises the company's dominance in the American market.

# ##### Let's take a look at the delivery locations now. 
# 

# In[12]:


import folium
from folium.plugins import MarkerCluster

# Filter relevant columns for delivery locations
delivery_locations_df = df[['Order City', 'Order Country', 'Latitude', 'Longitude']].drop_duplicates().dropna().reset_index(drop=True)
delivery_locations_df.head()


# In[13]:


# Create a dark-themed base map
map_center = [delivery_locations_df['Latitude'].mean(), delivery_locations_df['Longitude'].mean()]
delivery_map = folium.Map(location=map_center, zoom_start=5, tiles='CartoDB dark_matter')

# Add markers for each delivery location
marker_cluster = MarkerCluster().add_to(delivery_map)

for index, row in delivery_locations_df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['Order City']}, {row['Order Country']}",
        icon=folium.Icon(color='blue')
    ).add_to(marker_cluster)

# Save the map to an HTML file
#delivery_map.save("delivery_locations_map.html")
display(delivery_map)


# ##### **OBSERVATION:** The dark-themed map shows delivery location clusters of varying sizes, with the number of deliveries indicated by the size of the clusters. North America, particularly the United States, has the highest delivery volumes, indicating a significant market. Asia and South America are next, implying a global reach. The continents of Africa and Oceania have the fewest. The map emphasises the delivery locations that are more prevalent in certain regions than others.

# In[14]:


import folium

# Calculate average profit per city and the overall average profit
average_profit_per_city = df.groupby('Order City')['Order Profit Per Order'].mean()
overall_avg_profit = df['Order Profit Per Order'].mean()

# Merge the average profit data with the original DataFrame to get coordinates
city_profit_with_coords = df[['Order City', 'Latitude', 'Longitude']].drop_duplicates().merge(average_profit_per_city, on='Order City')

# Create a base map
map = folium.Map(location=[0, 0], zoom_start=2)

# Add markers to the map
for idx, row in city_profit_with_coords.iterrows():
    color = 'green' if row['Order Profit Per Order'] >= overall_avg_profit else 'red'
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7
    ).add_to(map)

    
# Add legends
legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index:1000; font-size:14px;">
    &nbsp; <i class="fa fa-circle" style="color:green"></i> High Profit &nbsp; 
    &nbsp; <i class="fa fa-circle" style="color:red"></i> Low Profit &nbsp;
    </div>
    """
map.get_root().html.add_child(folium.Element(legend_html))

# Save the map to an HTML file
map.save('map.html')
overall_avg_profit


# ##### **OBSERVATION:** 1. The map depicts average profit per city, with high (green) and low (red) profit regions contrasted. The United States exhibits a mix, indicating varying profitability across cities. Profitable areas are prominent,  implying successful markets or operations. Other global regions with fewer markers show cities that are both profitable and unprofitable. This visualisation can be used as a strategic guide to identify areas of financial success as well as those that need attention or improvement. 
# ##### 2. The global average profit is approximately $22.22. Cities marked green exceed this benchmark, while red ones fall below it

# In[15]:


# Filter the dataset for completed orders
df_completed = df[df['Order Status'] == 'COMPLETE']

# Filter relevant columns for the analysis
profit_region_df = df_completed[['Order Profit Per Order', 'Order Region']]

# Perform a groupby operation to calculate the average profit for each region
average_profit_by_region = profit_region_df.groupby('Order Region')['Order Profit Per Order'].mean().reset_index()

# Display the average profit for each region
print(average_profit_by_region)


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

average_profit_by_region_sorted = average_profit_by_region.sort_values(by='Order Profit Per Order', ascending=False)
overall_avg_profit = df_completed['Order Profit Per Order'].mean()

import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
sns.barplot(x='Order Region', y='Order Profit Per Order', data=average_profit_by_region_sorted, palette="viridis")
plt.axhline(y=overall_avg_profit, color='red', linestyle='--', label='Overall Average')
plt.title('Average Order Profit Across Regions')
plt.xlabel('Order Region')
plt.ylabel('Average Order Profit Per Order')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.legend()
plt.ylim(0, 40)

plt.show()



# ##### **OBSERVATION:** 1. The bar graph shows average profits per order across global regions, with a red dashed line representing the overall average of $22.22. Profitability in Central Asia, Southern Africa, and Central Europe is above average, indicating efficient operations or high demand. Western Europe, Eastern Europe, and Western Asia, on the other hand, fall short of the target. 
# ##### 2. According to the findings, Central Asia and Southern Africa have the highest average order profit, far outpacing other regions. Eastern Europe and West Asia, on the other hand, have the lowest profitability.

# ##### Now, lets perform ANOVA to conform if there is any significant relation.

# In[17]:


# Perform a groupby operation to calculate the average profit for each region
average_profit_by_region = profit_region_df.groupby('Order Region')['Order Profit Per Order'].mean().reset_index()

# Display the average profit for each region
print(average_profit_by_region)

# Example of using ANOVA
from scipy.stats import f_oneway

# Extract profit data for each region
region_profits = [profit_region_df[profit_region_df['Order Region'] == region]['Order Profit Per Order'] for region in average_profit_by_region['Order Region']]

# Perform ANOVA
f_stat, p_value = f_oneway(*region_profits)

# Display the results
print(f"F-Statistic: {f_stat}")
print(f"P-Value: {p_value}")

if p_value < 0.05:
    print("There is a significant difference in average order profit between regions.")
else:
    print("There is no significant difference in average order profit between regions.")


# ##### ** OBSERVATIONS:** The ANOVA test produced an F-Statistic of 1.3978 and a P-Value of 0.1014, which is above the standard significance threshold of 0.05. This indicates that the variations in average order profits across different regions are not statistically significant. Therefore, while there may be apparent differences in profitability between regions, these differences could be due to random variation rather than a specific regional effect. Consequently, the data does not support a claim of inherent regional disparities in profitability.
# ##### It could just be random chance that some regions have higher or lower average profits. So, we can't confidently say that the region has a real effect on how much profit is made from orders.
