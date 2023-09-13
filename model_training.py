#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from datetime import timedelta,datetime
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,FunctionTransformer
from sklearn.linear_model import ElasticNet,LinearRegression
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
import warnings
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer


# In[2]:


df=pd.read_excel('output_all_students_Train_v9.xlsx')


# In[3]:


warnings.filterwarnings("ignore")
def prepare_data(data):            
    if data is None:
        return None

    def only_float(price):
        try:
            x = re.findall(r'[-+]?\d*\.\d+|\d+', price)
            if x:
                return float(x[0])
            else:
                return np.nan
        except:
            return np.nan

    def only_int(price):
        x=''
        try:
            if re.search('[,]',price):
                price_lst=price.split(',')
            else:
                price_lst=price.split()
            for i in price_lst:
                x+=i
            x=re.findall('[0-9]+',x)
            x=float(x[0])
            return x
        except:
            return np.nan

    def no_punc(text):
        # Function to remove punctuation from a string
        import string
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)

    def format_enter(enter):
        if re.search(r'[0-9]+-[0-9]',enter):
            enter=datetime.strptime(enter, '%Y-%m-%d %H:%M:%S').date()
            diff=enter-datetime.now().date()
            if diff.days<183:
                enter='less_then_6_months'
            elif diff.days>183 and diff.days<365:
                enter='months_6_12'
            else:
                enter='above_year'
        elif enter=='גמיש':
            enter='flexible'
        elif enter=='מיידי':
            enter='less_then_6_months'
        else:
            enter='not_defined'
        return enter

    def check_values(x):
        values_to_check = ['True', 'יש מעלית', 'יש', 'yes','כן','yes','יש חנייה','יש מחסן','יש מיזוג אויר','יש מרפסת','יש ממ"ד','נגיש','נגיש לנכים']
        if x in values_to_check:
            return 1
        else:
            return 0 
        
        
        
    filtered_rows = data['price'].notnull() &  ~data['price'].astype(str).str.contains(r'\d')
    # Assign filtered values to the 'price' column
    data.loc[filtered_rows, 'price'] = np.nan  # or any desired value or operation
    #Drop Nan in price
    
    data.loc[:, 'price'] = data['price'].astype('str').apply(only_int).astype('float')
    # Rename columns by removing spaces
    data.rename(columns=lambda x: x.replace(' ', ''), inplace=True) 
    
    #Shuffle the rows randomly 
    data = data.sample(frac=1, random_state=42).reset_index(drop=True).copy()
    
    #Add the Functions
    data.loc[:, 'Area'] = data['Area'].astype('str').apply(only_int).astype('float')
    data.loc[:, 'publishedDays'] = data['publishedDays'].astype('str').apply(only_int).astype('float')
    data.loc[:, 'room_number'] = data['room_number'].astype('str').apply(only_float).astype('float')
    data.loc[:, 'floor'] = data['floor_out_of'].astype('str').apply(only_int).astype('float')
    data.loc[:, 'num_of_images'] = data['num_of_images'].astype('str').apply(only_int).astype('float')
    data.loc[:, 'total_floors'] = data['floor_out_of'].astype('str').apply(lambda x: x.split()[-1] if x.split()[-1].isdigit() else    0).astype('float')
    
    data.loc[:, 'type'] = data['type'].astype('str').apply(no_punc)
    data.loc[:, 'Street'] = data['Street'].astype('str').apply(no_punc)
    data.loc[:, 'city_area'] = data['city_area'].astype('str').apply(no_punc)
    data.loc[:, 'furniture'] = data['furniture'].astype('str').apply(no_punc)
    data.loc[:, 'description'] = data['description'].astype('str').apply(no_punc)
    
    
    data.loc[:, 'condition'] = data['condition'].astype('str')
    data.loc[:, 'City'] = df['City'].str.replace(r'^\s+|\s+$', '', regex=True).astype('str')
    data.loc[:, 'City'] = df['City'].str.replace('נהרייה', 'נהריה')

   
    data.loc[:, 'hasElevator'] = data['hasElevator'].astype('str').apply(check_values)
    data.loc[:, 'hasParking'] = data['hasParking'].astype('str').apply(check_values)
    data.loc[:, 'hasStorage'] = data['hasStorage'].astype('str').apply(check_values)
    data.loc[:, 'hasAirCondition'] = data['hasAirCondition'].astype('str').apply(check_values)
    data.loc[:, 'hasBalcony'] = data['hasBalcony'].astype('str').apply(check_values)
    data.loc[:, 'hasMamad'] = data['hasMamad'].astype('str').apply(check_values)
    data.loc[:, 'handicapFriendly'] = data['handicapFriendly'].astype('str').apply(check_values)
    data.loc[:, 'hasBars'] = data['hasBars'].astype('str').apply(check_values)
    
    data.loc[:, 'entrance_date'] = data['entranceDate'].astype('str').apply(format_enter)
    
    #Fill missing values
    missing_values = [np.nan, None, '', '-', 'NaN','nan','None',None]
    # Fill missing values in 'city_area' column based on 'Street'
    data['city_area'] = data.apply(lambda row: row['Street'] if pd.isnull(row['city_area']) or row['city_area'] in missing_values else row['city_area'], axis=1)
    # Fill missing values in 'Street' column based on 'city_area'
    data['Street'] = data.apply(lambda row: row['city_area'] if pd.isnull(row['Street']) or row['Street'] in missing_values else row['Street'], axis=1)
    
    
    #Drop unnecessary columns
    data = data.drop(columns=['floor_out_of', 'number_in_street', 'entranceDate', 'num_of_images','publishedDays', 'entrance_date', 'furniture' ,'handicapFriendly', 'hasStorage', 'hasBars', 'hasParking', 'hasBalcony','hasElevator']).copy()
    
    return data

df  = prepare_data(df)


# In[4]:


#קריאה ופיצול הדאטה למבחן ולאימון 
df=pd.read_excel('Dataset_for_test.xlsx')
test = prepare_data(df)
X_test = test.drop('price', axis=1)
y_test = test['price']

df=pd.read_excel('output_all_students_Train_v9.xlsx')
df  = prepare_data(df)

#טיפול בערכים חריגים 
df= df[df['price'] > 0]
#df= df[df['room_number'] < 10]
#df= df[df['Area'] < 700]

X_train = df.drop('price', axis=1)
y_train = df['price']


# In[7]:


from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import issparse

pipeline = Pipeline([
    
    # Preprocessing
    ('preprocessing', ColumnTransformer([
        # Numeric Features
        ('numeric_features', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), [ 'Area','room_number','floor','total_floors',]),
        
        # Boolean Features
        ('boolean_features', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ]), ['hasMamad', 'hasAirCondition']),
        
        # Categorical Features
        ('categorical_features', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False))
        ]), ['City','city_area','description', 'Street', 'condition', 'type']),
        
        # Text Feature
        ('text_feature', CountVectorizer(stop_words=None, max_features=5000000, ngram_range=(1, 4)), 'description')
    ], remainder='passthrough')),
    
    #TruncatedSVD
    #('TruncatedSVD', TruncatedSVD(n_components=900) ),
    
    # Model
    ('model', ElasticNet(alpha=0.00035938136638046257, l1_ratio=0.9))
])
pipeline
pipeline.fit(X_train, y_train)


# In[11]:


import pickle
pickle.dump(pipeline, open("trained_model.pkl", "wb"))


# In[9]:


y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('MSE with best parameters:', len(str(int(mse))) )
print('MSE with best parameters:', mse**0.5 )


# In[ ]:


# Set up KFold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Define the range of alpha and l1_ratio values for grid search
alphas = np.logspace(-4, 1, 10)

l1_ratios = np.arange(0.1, 1, 0.1)
# Define the parameter grid for grid search
param_grid = {
    'preprocessing__text_feature__stop_words': [None],
    'preprocessing__text_feature__max_features': [100, 500000],
    'preprocessing__text_feature__ngram_range': [(1, 1), (1, 3), (1, 4)],
    'preprocessing__text_feature__tokenizer': [BertTokenizer.from_pretrained("bert-base-multilingual-cased")],
    'model__alpha': alphas,
    'model__l1_ratio': l1_ratios,
    'pca__n_components': [100, 550,850,1200]  # Add different values for n_components
}


# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Fit the pipeline with different hyperparameter combinations
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print('Best Parameters:', best_params)

# Set the best parameters for all steps
pipeline.set_params(**best_params)

# Calculate MSE with the best parameters
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('MSE with best parameters:', mse)


# In[13]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select the numeric features for VIF calculation
numeric_features = ['room_number', 'Area', 'num_of_images', 'publishedDays', 'floor', 'total_floors', 'price']

# Convert numeric columns to appropriate data type
df[numeric_features] = df[numeric_features].astype(float)

# Handle missing values in numeric features
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

# Handle invalid values (e.g., inf, NaN) in numeric features
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=numeric_features, inplace=True)

# Calculate VIF for each numeric feature
vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_features
vif_data["VIF"] = [variance_inflation_factor(df[numeric_features].values, i) for i in range(len(numeric_features))]


boolean_features = ['hasElevator', 'hasParking', 'hasBars', 'hasStorage', 'hasAirCondition', 'hasBalcony', 'hasMamad']

# Convert boolean columns to appropriate data type
df[boolean_features] = df[boolean_features].astype(bool)
boolean_data = df[boolean_features]
# Calculate VIF for each boolean feature
vif_data_bool = pd.DataFrame()
vif_data_bool["Feature"] = boolean_features
vif_data_bool["VIF"] = [variance_inflation_factor(boolean_data.values, i) for i in range(len(boolean_features))]


print("Numeric Features:")
print(vif_data)
print("\nBoolean Features:")
print(vif_data_bool)


# In[37]:


# Filter out all warnings
warnings.filterwarnings("ignore")
import pandas as pd
import ppscore as pps
import seaborn as sns
import matplotlib.pyplot as plt

#df = df.drop(columns=['city_area']).copy()


# Calculate the PPS matrix
pps_matrix = pps.matrix(df)

# Extract the scores and create a DataFrame
pps_scores = pps_matrix[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')

# Create a heatmap to display the PPS matrix
plt.figure(figsize=(18,15))
sns.heatmap(pps_scores, annot=True, cmap='Oranges', vmin=0, vmax=1, square=True)
plt.title("PPS Matrix")
plt.show()
# Re-enable warnings (optional)
warnings.filterwarnings("default")


# In[7]:


import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Calculate the Pearson's correlation matrix
correlation_matrix = df.corr()

# Create a heatmap to display the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True)
plt.title("Pearson's Correlation Coefficient Matrix")
plt.show()

