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



#%%
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming your DataFrame is named 'df'
# sns.set(style="whitegrid")
# plt.figure(figsize=(10, 6))

# # Plotting the count of each shipment mode
# sns.countplot(x='Shipping Mode', data=df, palette='viridis')

# plt.title('Count of Shipment Modes')
# plt.xlabel('Shipping Mode')
# plt.ylabel('Count')
# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming your DataFrame is named 'df'

# # Set the style of seaborn
# sns.set(style="whitegrid")

# # Create a boxplot to visualize the distribution of 'Benefit per order'
# plt.figure(figsize=(12, 8))
# sns.boxplot(x='Shipping Mode', y='Benefit per order', data=df)
# plt.title('Distribution of Benefit per order by Shipping Mode')
# plt.xlabel('Shipping Mode')
# plt.ylabel('Benefit per order')
# plt.show()

# # Create a violin plot for a detailed view of the distribution
# plt.figure(figsize=(12, 8))
# sns.violinplot(x='Shipping Mode', y='Benefit per order', data=df)
# plt.title('Distribution of Benefit per order by Shipping Mode')
# plt.xlabel('Shipping Mode')
# plt.ylabel('Benefit per order')
# plt.show()

# # Create a histogram to visualize the overall distribution
# plt.figure(figsize=(12, 8))
# sns.histplot(df['Benefit per order'], bins=30, kde=True)
# plt.title('Distribution of Benefit per order')
# plt.xlabel('Benefit per order')
# plt.ylabel('Frequency')
# plt.show()



#%%


import seaborn as sns
import matplotlib.pyplot as plt



# Assuming your DataFrame is named 'df'
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

#%%
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

# %%
correlation = df.groupby('Shipping Mode')[['Sales per customer', 'Benefit per order']].corr().iloc[0::2]['Benefit per order'].reset_index()
correlation
# %%
#First Class:
#The correlation coefficient between "Benefit per order" and "Sales per customer" for First Class shipping mode is 0.159849. This positive correlation indicates a moderate positive relationship between the two variables.

#Same Day:
#For Same Day shipping mode, the correlation coefficient is 0.139488. This positive correlation suggests a moderate positive relationship between the benefit per order and sales per customer.

#Second Class:
#The correlation coefficient for Second Class shipping mode is 0.117577. This positive correlation indicates a positive relationship, but it is weaker compared to First Class and Same Day.

#Standard Class:
#The correlation coefficient for Standard Class shipping mode is 0.131890. Similar to Same Day, there is a moderate positive relationship between benefit per order and sales per customer.

#In summary, all shipping modes show positive correlations, indicating that as sales per customer increase, the benefit per order also tends to increase. However, the strength of the correlation varies, with First Class having the strongest positive correlation.

#%%

# Assuming your DataFrame is named 'df'
# Extracting data for each shipping mode
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


#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'df'

# Create a box plot
sns.boxplot(x='Shipping Mode', y='Benefit per order', data=df)
plt.title('Distribution of Benefit per order by Shipping Mode')
plt.show()

# %%
#The ANOVA result you provided indicates that the p-value is 0.1147, which is greater than the typical significance level of 0.05. Therefore, based on this result, you fail to reject the null hypothesis. This suggests that there is not enough evidence to conclude that the means of "Benefit per order" are significantly different across the different shipping modes.

#In other words, the ANOVA test does not provide sufficient evidence to suggest that there is a significant difference in the average "Benefit per order" between the shipping modes. Keep in mind that the interpretation is based on the chosen significance level, and different levels may lead to different conclusions.

#If you have any more questions or if there's anything else you'd like to explore in your data, feel free to let me know!


#%%
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# Combine the data and create corresponding labels
data = pd.concat([first_class, same_day, second_class, standard_class])
labels = ['First Class'] * len(first_class) + ['Same Day'] * len(same_day) + ['Second Class'] * len(second_class) + ['Standard Class'] * len(standard_class)

# Perform Tukey's HSD test
tukey_results = pairwise_tukeyhsd(data, labels, alpha=0.05)

# Display the results
print(tukey_results)

# %%
#The results of the Tukey HSD test are presented in the table
#and it appears that none of the pairwise differences in means between the shipping modes are statistically significant
# The "reject" column indicates whether the null hypothesis of equal means is rejected
# Since all entries in the "reject" column are marked as "False," it suggests that there is no significant difference between the means of "Benefit per order" for any pair of shipping modes.

#In summary, both the ANOVA test and the Tukey HSD test indicate that there is not enough evidence to conclude that the means of "Benefit per order" are significantly different among the different shipping modes.


#%%



# %%
# Importing required libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Assuming the dataset is stored in a variable df

# Encoding categorical variables
label_encoder = LabelEncoder()
df['Shipping Mode'] = label_encoder.fit_transform(df['Shipping Mode'])

# Splitting the data into features and target variable
X = df[['Benefit per order']]
y = df['Shipping Mode']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the random forest model
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Making predictions
y_pred = random_forest_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the random forest model: {:.2f}%".format(accuracy * 100))

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#%%
from sklearn.metrics import classification_report

# Classification Report for Random Forest
classification_rep_rf = classification_report(y_test, y_pred)
print("Classification Report - Random Forest:")
print(classification_rep_rf)



#%%
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Convert labels to one-hot encoding
y_test_bin = label_binarize(y_test, classes=range(len(set(y_test))))

# Get predicted probabilities for each class
y_prob_rf = random_forest_model.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

class_names = ['First Class', 'Same Day', 'Second Class', 'Standard Class']

for i in range(len(set(y_test))):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob_rf[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC Curve for each class with class names
plt.figure(figsize=(8, 6))
for i in range(len(set(y_test))):
    plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve {} (area = {:.2f})'.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()



# %%


#%%
# # Adding the predicted classes to the DataFrame
df['Predicted Shipping Mode'] = random_forest_model.predict(X)

# Scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Predicted Shipping Mode', y='Benefit per order', data=df, hue='Shipping Mode', palette='viridis', alpha=0.6)
plt.title('Scatter Plot of Predicted Shipping Mode vs. Benefit per order')
plt.xlabel('Predicted Shipping Mode')
plt.ylabel('Benefit per order')
plt.legend()
plt.show()
#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming your Random Forest model is named 'random_forest_model'

# Make predictions on the test set
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


# %%
from xgboost import XGBClassifier


# Handle missing values and drop unnecessary column

# Encode categorical variable 'Shipping Mode'
label_encoder = LabelEncoder()
df['Shipping Mode'] = label_encoder.fit_transform(df['Shipping Mode'])

# Select features and target variable
X = df[['Benefit per order']]
y = df['Shipping Mode']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the XGBoost model
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_xgb)
mse = mean_squared_error(y_test, y_pred_xgb)
print("Accuracy of the XGBoost model: {:.2f}%".format(accuracy * 100))
print("Mean Squared Error:", mse)



#%%
from sklearn.metrics import classification_report

# Assuming 'y_test' and 'y_pred_xgb' are already defined

# Generate a classification report
class_report = classification_report(y_test, y_pred_xgb)

# Print the classification report
print("Classification Report for XGBoost Model:\n")
print(class_report)









#%%
# Feature Importance Plot for XGBoost
feature_importance_xgb = xgb_model.feature_importances_
features_xgb = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_xgb, y=features_xgb)
plt.title('Feature Importance - XGBoost')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()



#%%
from sklearn.model_selection import learning_curve
import numpy as np
# Plot Learning Curve for XGBoost
train_sizes, train_scores, test_scores = learning_curve(xgb_model, X, y, cv=5, scoring='accuracy')

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Validation Score')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.title('Learning Curve - XGBoost')
plt.legend()
plt.show()



