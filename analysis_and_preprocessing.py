
import numpy as np 
import pandas as pd 
import statistics 
from sklearn.preprocessing import LabelBinarizer 


def create_df(data_path):
    excel_path = data_path + "\housing.csv"  
    df = pd.read_csv(excel_path)  
    return df  


def nan_columns(df):
    nancolumns = []  
    columns_Name = df.columns.values  
    total_rows_num = max(df.count())  
    for col in columns_Name:  
        if df[
            col].count() < total_rows_num:  
            nancolumns.append(col)  
    return nancolumns  


def categorical_columns(df):
    columns_Name = df.columns.values  
    catcolumns = []  
    all_list = [] 
    for col in columns_Name:
        categorial_values = df[col].get_values()  
        for i in categorial_values:  
            if type(i) == str:  
                all_list.append(col)  
        catcolumns = list(set(all_list)) 
    return catcolumns 


def replace_missing_features(df, nancolumns):
    new_df1 = df  
    for col in nancolumns:  
        median_number = statistics.median(df[col].values)  
        update_df1 = new_df1[col].fillna(median_number)  
        new_df1.update(update_df1)  
    print new_df1.head()
    print df.head()
     
     
     
def cat_to_num(new_df1, catcolumns):
    new_df2 = new_df1  
    OneHot_Encoder = LabelBinarizer()  
    for col in catcolumns:  
        new_df2 = new_df2.drop(col, axis=1)  
        one_Hot_Encoded = OneHot_Encoder.fit_transform(new_df1[col].get_values()) # One Hot Encoding
        clas = OneHot_Encoder.classes_  
        for i in range(len(new_df1[col].get_values())):
            new_df1[col].get_values()[i] = one_Hot_Encoded[i].tolist() 
        old_mat = new_df1[col].get_values().tolist() 
        new_mat = np.transpose(old_mat)
        for j in range(len(new_mat)):
            new_df2[clas[j]] = new_mat[j, :] 
    return new_df2 


def standardization(new_df2, labelcol):
    new_df3 = new_df2 
    columns_Name = new_df3.columns.values 
    for col in columns_Name:
        if labelcol == col: 
            pass
        else: 
            data = new_df3[col].values  
            mean = new_df3[col].mean()  
            standard_deviation = new_df3[col].std()  
            data_standarization = (data - mean) / standard_deviation 
            new_df3[col] = data_standarization  
    return new_df3 


def my_train_test_split(new_df3, labelcol, test_ratio):
    columns_Name = new_df3.columns.values 
    np.random.seed(0)  
    for col in columns_Name:
        if labelcol == col: 
            pass
        else:
            data_X = new_df3 
            data_y = new_df3[labelcol] 
            rand_ind_X = np.random.permutation(len(data_X)) 
            rand_ind_y = np.random.permutation(len(data_y))  
            X_test_data = int(len(data_X) * test_ratio) 
            y_test_data = int(len(data_y) * test_ratio) 
            X_test_ind = rand_ind_X[:X_test_data] 
            y_test_ind = rand_ind_y[:y_test_data] 
            X_train_ind = rand_ind_X[X_test_data:] 
            y_train_ind = rand_ind_y[y_test_data:] 
            y_train = new_df3[labelcol].iloc[y_train_ind] 
            y_test = new_df3[labelcol].iloc[y_test_ind]  
            new_df3 = new_df3.drop(labelcol, axis=1)  
            X_train = new_df3.iloc[X_train_ind]   
            X_test = new_df3.iloc[X_test_ind]  
            return X_train, X_test, y_train, y_test 

def main(dataPath, testRatio, labelColumn): 
    df = create_df(dataPath)  
    nan_col = nan_columns(df)  
    cat_col = categorical_columns(df)  
    df=replace_missing_features(df, nan_col)  
    cat_num = cat_to_num(df, cat_col)  
    stdz = standardization(cat_num, labelColumn)  
    X_train, X_test, y_train, y_test =  my_train_test_split(stdz, labelColumn, testRatio) 
    return X_train, X_test, y_train, y_test 



