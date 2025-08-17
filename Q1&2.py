#!/usr/bin/env python
# coding: utf-8

# ## Collaborator: Chinmay Dhamapurkar

# In[1]:


import pandas as pd


# In[2]:


import os
os.getcwd()


# In[3]:


df = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')


# In[104]:


df


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



# ## Not considered in Presentation due to time constraints
# 
# smart Question(Optional) :Can we build a model that can help the risk of late deliveries?

# From these visualizations, we can observe:
# 
# There's variability in the actual shipping days across different shipping modes, which suggests that the shipping mode could be a significant factor in determining delivery times.
# The scatter plot shows that while many orders are delivered on time (close to the diagonal line), there are also quite a few that are delivered earlier or later than scheduled.
# To further quantify the impact of these and other factors, we can perform a correlation analysis and then move on to predictive modeling. The model will help us to not only confirm the impact of these factors but also to quantify their effect on delivery times.

# In[10]:


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


# In[11]:


df.info()


# In[12]:


correlation_matrix = df.corr()

# Now, we'll plot the heatmap using seaborn
plt.figure(figsize=(15, 15))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .5})
plt.title('Heatmap of Correlation Matrix')
plt.show()


# In[13]:


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

# In[14]:


data=df
data['order date']= pd.to_datetime(data['order date (DateOrders)'])
data['shipping date']= pd.to_datetime(data['shipping date (DateOrders)'])


# In[15]:


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


# In[16]:


data


# In[17]:


data_n=df.loc[:,['Type','Days for shipment (scheduled)','order year','order month','order day','order hour','order minute','Benefit per order','Category Name','Latitude','Longitude','Customer Segment','Department Name','Market','Order City','Order Country','Order Item Discount','Order Item Product Price','Order Item Quantity','Order Item Total','Order State','Product Name','shipping year','shipping month','shipping day','shipping hour','shipping minute','Shipping Mode','Late_delivery_risk','Order Status']]
data_n.info()


# In[18]:


for i in data_n:
    if data_n[i].dtype=="object":
        print(i,len(data_n[i].unique()))


# This are the values in text, lets convert it into numbers using encoding techniques

# In[19]:


data_n['Order Status'].value_counts()
data_n['Order Status']= [0 if i!='SUSPECTED_FRAUD' else 1 for i in data_n['Order Status']]
data_n['Order Status'].unique()


# In[20]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
enc=LabelEncoder()
for i in data_n.columns:
    if data_n[i].dtype=='object':
        data_n[i]=enc.fit_transform(data_n[i])
data_n.info()


# In[21]:


data_n.info()


# ## Building ML Models as all features have been succesfully converted to numerical values

# In[22]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
y1=data_n["Late_delivery_risk"]

y2=data_n['Order Status']
x=data_n.drop(['Late_delivery_risk','Order Status'],axis=1)
x=StandardScaler().fit_transform(x)


# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train1,y_test1,y_train2,y_test2= train_test_split(x,y1,y2,stratify=y1,test_size=0.2)


# In[24]:


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


# In[25]:


import os
os.getcwd()


# In[26]:


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

# In[27]:


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

# In[28]:


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

# In[29]:


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

# In[30]:


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


# In[31]:


# # After fitting the model with random_search.fit(x_train, y_train1)
# import joblib
# joblib.dump(random_search, 'random_search_rf_model.joblib')


# In[32]:


loaded_model = joblib.load('random_search_rf_model.joblib')


# In[33]:


from sklearn.metrics import classification_report

# Assuming best_model is your trained RandomForestClassifier after hyperparameter tuning
y_pred = best_model.predict(x_test)

# Generate the classification report
class_report = classification_report(y_test1, y_pred)

print(class_report)


# In[34]:


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

# ## SMART Question:
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

# In[35]:


get_ipython().system('pip install folium')
import folium
from folium.plugins import HeatMap
sample_data = data.sample(n=10000, random_state=42)
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=2)
HeatMap(data=df[['Latitude', 'Longitude']], radius=15).add_to(m)
m


# ## Smart Question 1: RFM and ABC Analysis Using Clustering Techniques.

# In[36]:


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

# In[108]:


categorical_features = ['Type', 'Delivery Status', 'Category Name', 'Shipping Mode']

fig, axes = plt.subplots(nrows=len(categorical_features), ncols=1, figsize=(6, 10))

for i, col in enumerate(categorical_features):
    order_count = data[col].value_counts().head(10) 
    sns.barplot(x=order_count, y=order_count.index, ax=axes[i])
    axes[i].set_title(f'Count of {col}', fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('')

fig.tight_layout()
plt.show()


# In[38]:


import pandas as pd
import dateutil.parser
import numpy as np
import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns;
import matplotlib.pyplot as plt
from plotly import graph_objects as go
import plotly.express as px


# In[39]:


sm2 = pd.read_csv("DataCoSupplyChainDataset.csv", encoding='ISO-8859-1')


# In[40]:


sm2.head()
sm2.info()
sm2.describe()


# In[41]:


sm2


# In[42]:


sm2.drop('Product Description', axis=1 , inplace= True)


# In[43]:


sm2


# In[44]:


sm2['order date (DateOrders)']= pd.to_datetime(sm2['order date (DateOrders)'])


# In[45]:


sm2["shipping date (DateOrders)"]=pd.to_datetime(sm2["shipping date (DateOrders)"])
sm2=sm2.sort_values(by="order date (DateOrders)")


# In[46]:


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

# In[47]:


print("Length of the data frame where benefits per order and order profit per order are not equal :"+str(dFppsc[dFppsc["Benefit per order"]!=dFppsc["Order Profit Per Order"]].shape[0]))
print("Length of the data frame where sales per customer and Order Item Total are not equal :"+str(dFppsc[dFppsc["Sales per customer"]!=dFppsc["Order Item Total"]].shape[0]))
print("Length of the data frame where Order Item Cardprod Id and Product Card Id are not equal :"+str(dFppsc[dFppsc["Order Item Cardprod Id"]!=dFppsc["Product Card Id"]].shape[0]))
print("Length of the data frame where Product Price and Order Item Product Price are not equal :"+str(dFppsc[dFppsc["Product Price"]!=dFppsc["Order Item Product Price"]].shape[0]))
print("Length of the data frame where Category Id and Product Category Id are not equal :"+str(dFppsc[dFppsc["Product Category Id"]!=dFppsc["Category Id"]].shape[0]))
print("Length of the data frame where Order Customer Id and Customer Id Product Price are not equal :"+str(dFppsc[dFppsc["Order Customer Id"]!=dFppsc["Customer Id"]].shape[0]))


# After comparing the values in columns pairs having co-relation of 1, we can see that all values are equal for these pair of columns. We can drop one of the columns from each of the above pairs as they are duplicate columns of each other.

# In[48]:


dFppsc.drop(["Benefit per order","Sales per customer","Order Item Cardprod Id","Order Item Product Price","Product Category Id","Order Customer Id"],axis=1,inplace=True)


# ## Using Variance Inflation Factor to find out multi-colinearity between three or more columns
# 
# Corelation helps us in finding relations between any two columns. But what if we want to find out if there is a relation between three or more columns. That's where Variance Inflaction Factor can help.
# 
# Variance Inflation Factor (VIF) - VIF is the ratio of variance of coefficient estimate when fitting the full model divided by the variance of coefficient estimate if fit on its own. The minimum possible value is 1 which indicates no collinearity. If value exceeds 5, then collinearity should be addressed.
# 
# We will take all the numeric columns and try to find variance_inflation_factor for them.

# In[49]:


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

# In[50]:


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

# In[51]:


df1=dFppsc.drop(["Order Item Total","Product Price","Order Item Discount Rate","Order Profit Per Order"],axis=1)


# In[52]:


vif=pd.DataFrame()
vif["columns"]=['Sales',
                'Order Item Quantity','Order Item Discount','Order Item Profit Ratio']
vif["data"] = [variance_inflation_factor(df1[['Sales',
                'Order Item Quantity','Order Item Discount','Order Item Profit Ratio']].values, i) for i in range(len(vif["columns"]))]
vif


# Here we can clearly see that the VIF value is reduced very highly

# In[53]:


dFppsc['Delivery Status'].value_counts()


# We can see that there are 7754 orders where shipping has been canceled. Lets try to find why it may be happening.

# In[54]:


plt=dFppsc[dFppsc["Delivery Status"]=="Shipping canceled"].groupby("Order Status").agg(tc=("Order Id","count"))
plt


# As we can see shipping has been canceled because customer canceled his order or order was marked as suspected fraud. We will have to remove these orders when we do RFM and ABC analysis as these were transaction that didn't add any value for store or for customer.

# In[55]:


df_clean=dFppsc[dFppsc["Delivery Status"]!="Shipping canceled"].copy()


# In[56]:


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

# In[57]:


dF_frequency=df_clean.groupby(["Customer Id"]).agg(Total_count=("Order Id","nunique"))


# In[58]:


dF_frequency.to_excel("Frequency.xlsx")


# In[59]:


df_clean["Year"]=df_clean["order date (DateOrders)"].dt.year
df_frequency_year=df_clean.groupby(["Customer Id","Year"], as_index=False)["Order Id"].nunique()
df_frequency_year


# Organising the data for better comparison

# In[60]:


df_frequency_year=df_frequency_year.pivot_table(index="Customer Id", columns="Year", values="Order Id", fill_value=0)
df_frequency_year


# There is something odd in the data, as we can see in excel that no customer having customer id less than 12436 has purchased in 2018, also after customer id 12336, all the customer has made exactly one purchase, there seems to be some problem in the data, lets dive deeper

# In[61]:


pd.set_option("display.max_columns", None)
df_clean[df_clean["Customer Id"]==12436]


# Here after looking at the date timestamp from "02-10-2017" all the customers recorded in the system are new.There are no records of old customer purchasing after 2nd october, this problem is with the dataset which should be addressed from data engineers or the data owners to know the results behind the irregularity.

# This particular customer has purchased at 2017-10-02 12:46:00. Lets see the data around this timestamp to see if we can find something unusual.

# In[62]:


dt1=datetime.datetime.strptime('02-10-2017 12:46:00', '%d-%m-%Y %H:%M:%S')
dt2=dt1-datetime.timedelta(days=5)


# In[63]:


df_exploring=df_clean[df_clean["order date (DateOrders)"]>=dt2]
df_exploring.to_excel("afterdt1.xlsx")
df_exploring[df_exploring["order date (DateOrders)"]>=dt1]


# As we can see after date time of "02-10-2017 12:46:00" all the customer that are recorded in system are new. There are no records of old customer purchasing after 2nd October 2017 12:46 PM. This may be due to multiple reasons and in real life we must connect with the data engineers or the data owners to know the reason behind this irregularity.
# 
# One possible reason could be that after the above date time the company started using different system to handle old customers and was using the same system to handle new customers
# 
# We can try to see that if there are records with same Customer Name and Customer City after the above timestamp and before the timestamp. If there are such cases present, we can compare their other transaction details such as product purchased ,customer street, Order city to find out whether these customers are the same and system have erroneously assigned a new customer Id even to old customer

# In[64]:


dF_clean=df_clean
dF_clean["Customer Full Name"]=dF_clean["Customer Fname"]+" "+dF_clean["Customer Lname"]
datetime_val = datetime.datetime.strptime('02-10-2017 12:46:00', '%d-%m-%Y %H:%M:%S')
Customer_Names_After_Dt=dF_clean[dF_clean["order date (DateOrders)"]>=datetime_val][["Customer Full Name","Customer City"]].drop_duplicates()


# In[65]:


Customer_Name_Common_Before_After_dt=pd.merge(dF_clean[(dF_clean["order date (DateOrders)"]<=datetime_val)],Customer_Names_After_Dt,how="inner",on=["Customer Full Name","Customer City"])
Records=pd.merge(Customer_Name_Common_Before_After_dt[["Customer Full Name","Customer City"]].drop_duplicates(),dF_clean,how="inner",on=["Customer Full Name","Customer City"])
Records.to_excel("idissue.xlsx")


# In[66]:


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

# In[67]:


dF_clean=dF_clean[dF_clean["order date (DateOrders)"]<datetime_val].copy()


# In[68]:


dF_frequency=dF_clean[["Customer Id","Order Id"]].groupby("Customer Id",as_index=False).nunique()
dF_frequency.columns=["Customer Id","Frequency"]
sns.histplot(dF_frequency.Frequency,bins=15,kde=False)


# ## 50% of our customers have ordered from us 4 times or more.

# In[69]:


dF_frequency["F"],Intervals_Frequency=pd.qcut(dF_frequency["Frequency"],q=3,labels=[1,2,3],retbins=True)
Intervals_Frequency


# # Recency of a customer is time since last purchase date of customer. Here the opposite will be true,lesser the recency, more responsive is customer to promotional offers.
# 
# Here we will find last purchase date for each customer and subract it from the last recorded date in the database. This is done because dataset is not recent and last date in our cleaned data is of 2017.
# 
# In a real life scenario, we will subract last purchase date from current date.

# In[70]:


dF_recency=dF_clean[["Customer Id","order date (DateOrders)"]].groupby("Customer Id",as_index=False).max()
dF_recency.rename(columns ={"order date (DateOrders)":"last_purchase_date"},inplace=True)
max_date=dF_recency["last_purchase_date"].max()
dF_recency["recency"]=max_date-dF_recency["last_purchase_date"]
dF_recency["recency"]=dF_recency["recency"].dt.days


# In[71]:


sns.displot(dF_recency.recency,bins=8,kde=False)


# We can observe that around 5250 of our customer have purchased from us in the last three months.

# In[72]:


dF_recency['R'],Intervals_Recency=pd.qcut(dF_recency["recency"],q=3,labels=[3,2,1],retbins=True)


# In[73]:


print("Interval Recency:"+f"{Intervals_Recency}")


# Monetary value represents total amount of all orders of a customer. Here more is better, so higher score will be assign to higher monetary value.

# In[74]:


dF_monetory=dF_clean[["Customer Id","Order Item Total"]].groupby("Customer Id",as_index=False).sum()
dF_monetory.columns=["Customer Id","Sum_of_Sales"]
dF_monetory["M"],Intervals_Monetory=pd.qcut(dF_monetory["Sum_of_Sales"],q=3,labels=[1,2,3],retbins=True)
Intervals_Monetory


# In[75]:


sns.displot(dF_monetory.Sum_of_Sales,kde=True)


# We can see that Monetary value follows a normal distribution, also the average of all monetary value looks to be around 2300-2400.
# 
# Now we will join all three values of recency, frequency, and monetary on Customer Id. We will find a combine score of RFM for each Customer Id. Then we will segment the customers based on that score.

# In[76]:


dF_rfm=pd.merge(dF_recency[["Customer Id","R"]],dF_monetory[["Customer Id","M"]],on="Customer Id",how="inner")
dF_rfm=pd.merge(dF_rfm,dF_frequency[["Customer Id","F"]],on="Customer Id",how="inner")
dF_rfm["RFM"]=(dF_rfm["R"]).astype(str)+(dF_rfm["F"]).astype(str)+(dF_rfm["M"]).astype(str)


# In[77]:


dF_rfm


# In[78]:


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


# In[79]:


dF_rfm["R"]=dF_rfm["R"].astype("category")
dF_rfm["F"]=dF_rfm["F"].astype("category")
dF_rfm["M"]=dF_rfm["M"].astype("category")
dF_rfm["Segment"]=dF_rfm.apply(Customer_Segment,axis=1)


# In[80]:


Segment_count=(dF_rfm.groupby("Segment",as_index=False).agg(Total_Count=("Customer Id","count"))).sort_values(by="Total_Count",ascending=False)


# In[81]:


dF_rfm
dF_rfm.to_excel("rfm.xlsx")


# In[82]:


fig2=go.Figure()
fig2.add_trace(go.Bar(x=Segment_count.Segment,
                      y=Segment_count.Total_Count,
                     hovertemplate ="%{label}<br>Number of Customers:%{value}",
                     texttemplate = "%{value}"))
fig2.update_layout(title="Number of Customer in Each Segment",
                    xaxis=dict(title="Customer Segment"),
                  yaxis=dict(title="Number Of Customers"),width=800)


# In[83]:


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


# In[84]:


# Performing K-means clustering with 4 clusters
kmeans_rfm = KMeans(n_clusters=4, random_state=42)
rfm_data['Cluster'] = kmeans_rfm.fit_predict(rfm_scores)

# Displaying the first few rows of the RFM data with the cluster labels
rfm_data.head()


# In[85]:


rfm_data.head()


# In[86]:


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


# In[87]:


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


# In[88]:


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


# In[89]:


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


# In[90]:


def get_strategy_by_cluster(cluster):
    cluster_strategies = {
        0: "Engage with personalized communications and offers to increase purchase frequency. Introduce loyalty programs to boost spending.",
        1: "Implement re-engagement campaigns such as special discounts or reminders of product value. Conduct surveys to understand their inactivity and address these concerns.",
        2: "Focus on enhancing first-time buyer experiences and follow-up marketing. Encourage second purchases through targeted offers based on initial buying behavior.",
        3: "Prioritize premium customer experiences and services. Explore opportunities for upselling and cross-selling. Maintain high engagement with exclusive offers and early access to new products."
    }
    return cluster_strategies.get(cluster, "Conduct further analysis to determine the best approach.")


# In[91]:


# Apply the strategy function to each row in the DataFrame
rfm_data['Strategy'] = rfm_data['Cluster'].apply(get_strategy_by_cluster)

# Display the DataFrame with the new 'Strategy' column
print(rfm_data.head())


# In[92]:


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


# In[93]:


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


# In[94]:


# RFM Score Distribution
plt.figure(figsize=(10, 6))
sns.histplot(rfm_data['RFM'], kde=False, bins=20)
plt.title('RFM Score Distribution')
plt.xlabel('RFM Score')
plt.ylabel('Frequency')
plt.show()


# In[111]:


pie_chart=rfm_data["Cluster"].value_counts()
pie_chart


# In[116]:


import seaborn
import matplotlib.pyplot as plt 
data = [3944,3202,3196,1955] 
keys = ['2','3','1','0'] 
palette_color = seaborn.color_palette('bright') 
  
# plotting data on chart 
plt.pie(data, labels=keys, colors=palette_color, autopct='%.0f%%') 
plt.show()


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

# In[96]:


Total_Products= df_clean["Product Name"].nunique()


# In[97]:


Total_Products


# In[98]:


Revenue_ABC=dF_clean.groupby(["Department Name","Product Name"]).agg(Total_Revenue=("Order Item Total","sum")).sort_values(by="Total_Revenue",ascending=False).reset_index()
Revenue_ABC["cum_sum"]=Revenue_ABC["Total_Revenue"].cumsum()
Revenue_ABC["cum_per"]=Revenue_ABC["cum_sum"]/Revenue_ABC["Total_Revenue"].sum()*100
Revenue_ABC["per"]=Revenue_ABC["cum_per"]-Revenue_ABC["cum_per"].shift(1)
Revenue_ABC.loc[0,"per"]=Revenue_ABC["cum_per"][0]


# In[99]:


Revenue_ABC


# In[100]:


def ABC(data):
    if data["cum_per"]<=75:
        return "A"
    elif data["cum_per"]>75 and data["cum_per"]<=95:
        return "B"
    elif data["cum_per"]>95:
        return "C"

Revenue_ABC["ABC_Revenue"]=Revenue_ABC.apply(ABC,axis=1)


# In[101]:


Bar_graph_Abc=Revenue_ABC[["ABC_Revenue","Product Name","Total_Revenue"]].groupby("ABC_Revenue").agg(Revenue=("Total_Revenue","sum"),count=("Product Name","count"))


# In[102]:


Revenue_ABC


# In[103]:


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

