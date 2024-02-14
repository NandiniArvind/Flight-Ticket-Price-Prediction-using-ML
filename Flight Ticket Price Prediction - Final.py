#!/usr/bin/env python
# coding: utf-8

# # Preparing Data set

# Import the necessary libraries with aliases for further analysis, such as:

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# reading the dataset using pandas 

# In[2]:


data = pd.read_excel('S:\DS Corizo\Flight Ticket prediction\Data_Train.xlsx')


# data.head() only shows relevant data without repeating the headers

# In[3]:


data.head()


# To obtain fundamental dataset details, employ the info() method, which provides concise information such as data types, column counts, and memory usage.

# In[4]:


data.info()                  


# In[5]:


data.shape   #Rows, columns


# In[6]:


data.count()      #To count objects present in particular 


# In[7]:


data.dtypes


# The describe() method is applicable solely to numerical data, as it provides statistical summaries such as mean and standard deviation, making it unsuitable for non-numeric objects.

# In[8]:


data.describe()


# To identify null values, use either isna() or isnull(), then utilize sum() to count the occurrences of missing values in the dataset.

# In[9]:


data.isna().sum()                         #data.isnull().sum()


# Utilize pandas to create conditions for checking null values in a dataset, such as data.isnull() or data.isna()

# In[10]:


data[data['Route'].isna() | data['Total_Stops'].isna()]


#  Remove null values from the dataset using dropna(), and ensure permanency of changes by setting inplace=True.

# In[11]:


data.dropna(inplace=True)


# In[12]:


data.isna().sum()


# In[13]:


data.count()


# In[14]:


data.head()


# With our dataset prepared, we can now conduct feature engineering through Exploratory Data Analysis (EDA) to gain insights and enhance our understanding of the data's characteristics and relationships.

# # EDA & Feature Engineering

# 1. Duration 
# 2. Departure and Arrival time
# 3. Date of journey 
# 4. Total stops 
# 5. Additional info 
# 6. Airline
# 7. Source and destination
# 8. Route

# ### 1. Duration

# We can create a function called `convert_duration` to take the duration in hours:minutes format as input and return the corresponding duration in singular minutes format.

# In[15]:


def convert_duration(duration):
    if len(duration.split()) == 2:
        hours = int(duration.split() [0][:-1])
        minutes  = int(duration.split() [1][:-1])
        return hours * 60 + minutes 
    else:
        return int(duration[:-1])* 60

    


# In[16]:


data.head()


# In[17]:


data['Duration'] = data['Duration'].apply(convert_duration)
data.head()


# ### 2. Departure and Arrival Time

# To convert departure and arrival times from object type to datetime format, you can use the pd.to_datetime() function from the pandas library.

# In[18]:


data['Dep_Time'] = pd.to_datetime(data['Dep_Time'])
data['Arrival_Time'] = pd.to_datetime(data['Arrival_Time'])
data.dtypes


# we can extract the hours and minutes from the departure and arrival times by using the dt.hour and dt.minute accessor functions 

# In[19]:


data['Dep_Time_in_hours'] = data['Dep_Time'].dt.hour
data['Dep_Time_in_minutes'] = data['Dep_Time'].dt.minute
data['Arrival_Time_in_hours'] = data['Arrival_Time'].dt.hour
data['Arrival_Time_in_minutes'] = data['Arrival_Time'].dt.minute


# In[20]:


data.head()


# In[21]:


data.drop(['Dep_Time', 'Arrival_Time'],axis= 1, inplace = True)
data.head()


# ### 3. Date of Journey 

# Since the data spans from 2019, we can ignore the year and concentrate solely on the date and month information for analysis.

# In[22]:


data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'])
data.head()


# To streamline the data and reduce redundancy, we'll remove the year from the date of the journey and create new columns containing only the day and month information. 

# In[23]:


data['Day'] = data['Date_of_Journey'].dt.day
data['Month'] = data['Date_of_Journey'].dt.month
data.head()


# In[24]:


data.drop('Date_of_Journey', axis=1 , inplace= True)
data.head()


# ### 4. Total stops 

# In[25]:


data['Total_Stops'].value_counts()


# 
# To convert objects in the 'Total_stops' column into numerical data, we can use a mapping dictionary to assign numerical values to each category

# In[26]:


data['Total_Stops'] = data['Total_Stops'].map({
    'non-stop':0,
    '1 stop'  :1,
    '2 stops' :2,
    '3 stops' :3,
    '4 stops' :4
})


# In[27]:


data.head()


# ### 5. Additional Info 

# In[28]:


data['Additional_Info'].value_counts()


# To enhance data clarity and efficiency, we will discard the 'Additional_Info' column as it does not offer valuable information for analysis.

# In[29]:


data.drop('Additional_Info',axis=1, inplace= True)
data.head()


# ###### Following code is useful for filtering and extracting columns with object data type for further analysis or manipulation.

# In[30]:


data.select_dtypes(['object']).columns


# In[31]:


for i in ['Airline', 'Source', 'Destination', 'Total_Stops']:
    plt.figure(figsize = (15,6))
    sns.countplot(data = data, x = i)
    ax = sns.countplot( x= i , data = data.sort_values('Price', ascending = True))
    ax.set_xticklabels(ax.get_xticklabels() , rotation= 40 , ha = 'right' )
    plt.tight_layout()
    plt.show()
    print('\n\n')


# - The code iterates over a list of categorical columns: 'Airline', 'Source', 'Destination', and 'Total_Stops'.
# - For each column, it creates a count plot to visualize the frequency of each category.
# - The count plot is sorted based on the 'Price' column in ascending order.
# - X-axis labels are rotated by 40 degrees for better readability.
# - The plots are displayed with a tight layout, ensuring they don't overlap.
# - Two empty lines are printed between each plot for better separation and readability.

# ### 6. Airline
# 

# In[32]:


data['Airline'].value_counts()


# In[33]:


plt.figure(figsize = (15,6))
ax = sns.barplot(x = 'Airline' , y ='Price',data = data.sort_values('Price', ascending = False))
ax.set_xticklabels(ax.get_xticklabels() , rotation= 40 , ha = 'right' )
plt.tight_layout()
plt.show()


# - `plt.figure(figsize = (15,6))`: Sets the size of the figure to be displayed (width: 15 units, height: 6 units).
# - `ax = sns.barplot(x = 'Airline' , y ='Price',data = data.sort_values('Price', ascending = False))`: Creates a bar plot where the x-axis represents the airlines and the y-axis represents the prices, using the DataFrame 'data' sorted by price in descending order.
# - `ax.set_xticklabels(ax.get_xticklabels() , rotation= 40 , ha = 'right' )`: Sets the rotation of x-axis labels to 40 degrees and aligns them to the right for better readability.
# - `plt.tight_layout()`: Adjusts the layout of the plot to prevent overlapping.
# - `plt.show()`: Displays the plot.

# In[34]:


plt.figure(figsize = (15,6))
ax = sns.boxplot(x = 'Airline' , y ='Price',data = data.sort_values('Price' , ascending = False))
ax.set_xticklabels(ax.get_xticklabels() , rotation= 40 , ha = 'right' )
plt.tight_layout()
plt.show()


# - Boxplot is used to visualize the distribution of a continuous variable (in this case, 'Price') across different categories (in this case, 'Airline').
# - The dots in the boxplot represent outliers, which are data points that fall significantly far from the median of the distribution. They could indicate unusual or extreme values in the dataset.

# In[35]:


data.groupby('Airline').describe()['Price'].sort_values('mean', ascending = False)


# Above code is used to group the data by the 'Airline' column, then calculate descriptive statistics (mean, standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum) for the 'Price' column within each group. Finally, it sorts the results based on the mean price in descending order.

# In[36]:


Airline = pd.get_dummies(data['Airline'], drop_first=True).astype(int)
Airline.head()


# In[37]:


data = pd.concat([data,Airline], axis = 1)
data.head()


# In[38]:


data.drop('Airline', axis=1, inplace=True)
data.head()


# ### 7. Source and destination
# 

# In[39]:


list1 = ['Source' , 'Destination']
for i in list1:
    print(data[[i]].value_counts(),'\n')


# In[40]:


data = pd.get_dummies(data = data , columns= list1, drop_first= True , dtype=int)
data.head()


# ### 8. Route

# In[41]:


route = data[['Route']]
route.head()


# In[42]:


data['Total_Stops'].value_counts()


# In[43]:


route['Route_1'] = route['Route'].str.split('→').str[0]
route['Route_2'] = route['Route'].str.split('→').str[1]
route['Route_3'] = route['Route'].str.split('→').str[2]
route['Route_4'] = route['Route'].str.split('→').str[3]
route['Route_5'] = route['Route'].str.split('→').str[4]
route.head()


# In[44]:


route.fillna('None',inplace=True)


# In[45]:


route.head()


# In[46]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in range(1,6):
    col = 'Route_' + str(i)
    route[col] = le.fit_transform(route[col])
    
route.head()    


# In[47]:


route.drop('Route', axis = 1 , inplace = True)
route.head()


# In[48]:


data = pd.concat([data,route], axis = 1)
data.head()


# In[49]:


data.drop('Route' , axis = 1 , inplace = True)
data.head()


# In[ ]:





# ## Building Machine Learning Models

# In[50]:


temp_col = data.columns.to_list()
print(temp_col , '\n')

new_col = temp_col[:2] + temp_col[3:]
new_col.append(temp_col[2])
print(new_col,'\n')

data = data.reindex(columns = new_col)
data.head()


# The code creates a list of column names from the DataFrame 'data', then excludes the third column and appends it at the end. Finally, it reorders the DataFrame columns based on the modified list of column names, placing the third column at the end.

# In[51]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data = scaler.fit_transform(data)

data[0]


# - The code imports the StandardScaler class from the preprocessing module of scikit-learn.
# - It initializes a StandardScaler object named 'scaler'.
# - The 'fit_transform' method of the scaler is applied to the data, which computes the mean and standard deviation of each feature and scales the data accordingly.
# - Finally, it transforms the data using the calculated mean and standard deviation, returning the scaled data.
# - 'data[0]' retrieves the first row of the scaled data.

# In[52]:


from sklearn.model_selection import train_test_split as tts

x = data[:, :-1]
y = data[: , -1]

#The code splits the dataset 'data' into features (x) and target variable (y), where 'x' contains all columns except the last one, and 'y' contains only the last column.


# In[53]:


x_train , x_test , y_train , y_test = tts(x, y, test_size = 0.1, random_state = 69)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# This code splits the dataset 'x' and target variable 'y' into training and testing sets, with a test size of 10% and a random state set to 69 for reproducibility. It then prints the shapes of the resulting training and testing sets for both the features and the target variables.

# ### Linear Regression 

# The code utilizes scikit-learn's LinearRegression model to train and fit a linear regression algorithm on the training data (x_train) and corresponding target values (y_train).

# In[54]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train , y_train)


# Performance Metrics for Regression Models

# In[55]:


from sklearn.metrics import mean_squared_error , r2_score

def metrics(y_true , y_pred):
    print(f'RMSE:', mean_squared_error(y_true, y_pred)** 0.5)
    print(f'R_Squared value:', r2_score(y_true , y_pred ))
    
def accuracy(y_true , y_pred):
    errors = abs(y_true - y_pred)
    mape = 100*np.mean(errors/y_true)
    accuracy = 100 -mape
    return accuracy


# The code defines two functions for evaluating the performance of regression models:
# 1. **metrics**: It calculates and prints the root mean squared error (RMSE) and the R-squared value between the true target values (y_true) and the predicted values (y_pred).
# 2. **accuracy**: It computes the accuracy of the model predictions by calculating the mean absolute percentage error (MAPE) and then subtracting it from 100 to get the accuracy percentage.

# In[56]:


y_pred = model.predict(x_test)


# In[57]:


metrics(y_test, y_pred)


# In[58]:


accuracy(y_test, y_pred)


# ### Random Forest 

# In[59]:


from sklearn.ensemble import RandomForestRegressor

model_random_forest = RandomForestRegressor(n_estimators = 500 , min_samples_split= 3 )
model_random_forest.fit(x_train , y_train)


# This code uses scikit-learn's RandomForestRegressor, creating a forest of 500 decision trees. It then trains the model on the provided training data (x_train) and target values (y_train).

# In[60]:


pred_rf = model_random_forest.predict(x_test)


# In[61]:


metrics(y_test , pred_rf)


# In[62]:


accuracy(y_test, pred_rf)


# In[ ]:





# In[ ]:




