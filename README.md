# Exploratory Data Analysis and Machine Learning on Housing Dataset

This is a basic project aims to give a simple easy introduction to machine learning by perform exploratory data analysis on the Housing dataset and apply machine learning techniques to predict the median house value. The dataset is provided for testing and the functions implemented in this project are expected to work for different datasets as well.

## Dataset

The Housing dataset can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz. It contains information about housing in California such as population, median income, median house value, etc.

## Data Analysis & Pre-processing 

The following functions are implemented to analyze and pre-process the data:
1. create_df 
  takes the path of the csv file and turns it into a pandas dataframe
  
2. nan_columns 
  takes a dataframe as input and returns a list of names of columns that contain NaN values in it.
  
3. categorical_columns 
  takes a dataframe as input and returns a list of column names that contain categorical values in it.
  
4. replace_missing_features 
    takes a dataframe and a list of column names that contain NaN values as input. The function replaces all NaN values in the column with the median of this column’s values and returns this new dataframe as output.
    
5. cat_to_num 
    takes a dataframe and a list of categorical feature column names as input. It performs one-hot-encoding on categorical features, modifies the dataframe, and replaces its nominal features with their one-hot encoding representation.
    
6. standardization
  takes a dataframe and label column as input and scales all columns except the label column with standardization. The output is a new dataframe with scaled values.
  
7. my_train_test_split
    takes a dataframe, name of the label column, and percentage of test size as input. It splits the dataframe into X and y where y is the label column values and X is the feature values (all column values except the label column). Then, the function splits X and y into test and train sets as X_train, X_test, y_train, and y_test with the given test size. The output datatype is numpy array.
    
8. main
  The main function uses all the above functions and returns a train and test set at the end. The function takes the path of a csv file, test data percentage, and name of the label column as input. It first converts the csv data into a dataframe, fills NaN columns of this dataframe, finds in which column there are categorical values, fills up missing features in dataframes, converts categorical features into numerical format, scales all feature columns with standardization, and finally splits the final dataframe into train and test matrices according to the given label column and test ratio. It returns X_train, X_test, y_train, and y_test matrices.
  
## Model Evaluation 

In this section, a Linear Regression model is trained that learns to predict the label column according to the rest of the features in the dataset. The following functions are implemented:

1. model_evaluation

This function takes training instances (train_x) and labels (train_y) as numpy arrays. It creates a LinearRegression model with default parameters, trains the model with the train_x and train_y, and returns coef_ and intercept_ arrays of the model as output.

2. predict

This function takes an instance matrix, coef_ array, and intercept_ array as input. It creates a Linear Regression model and sets its coef_ and intercept_ parameters to input coef_ and intercept
