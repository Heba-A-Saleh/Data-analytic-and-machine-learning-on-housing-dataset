from sklearn.linear_model import LinearRegression # Linear Regression Model
import analysis_and_preprocessing as Train_Data # Importing the Main Function from the Analysis and Preprocessing File

def model_evaluation(X_train, y_train):
    Linear_Regression = LinearRegression() 
    Linear_Regression.fit(X_train, y_train)  
    coef_ = Linear_Regression.coef_  
    intercept_ = Linear_Regression.intercept_ 
    return coef_, intercept_ 
coef_,intercept_ = model_evaluation(Train_Data.X_train, Train_Data.y_train)

def predict(instance, coef_, intercept_):
    predictions = [] 
    data = Train_Data.X_test 
    if type(instance) == str:
        pred_result = coef_ * data + intercept_ 
        predictions.append(pred_result[instance].tolist()) 

    else:
        for col_name in instance: 
            pred_result = coef_ * data + intercept_ 
            predictions.append(pred_result[col_name].tolist()) 
    return predictions 




