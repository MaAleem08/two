import kagglehub
import pandas as pd , numpy as np , seaborn as sns 
from sklearn.preprocessing import StandardScaler , LabelEncoder ,OneHotEncoder ,FunctionTransformer
from sklearn.impute import KNNImputer , SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_classif ,SelectKBest ,chi2
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix ,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import pickle , joblib

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('diabetes.csv')

x_train,x_test,y_train,y_test = train_test_split(df.drop(columns = 'Outcome'), df.Outcome,test_size=0.25 , stratify=df.Outcome , random_state=123)


num_cols = df.drop(columns='Outcome').select_dtypes(include=['int64','float64']).columns
cat_cols = df.drop(columns='Outcome').select_dtypes(include=['object']).columns


num_pipe = Pipeline([
                            ('num_imputer' , KNNImputer(n_neighbors=5)),
                            ('scaler',StandardScaler()),
                            ('num_feature_selection',SelectKBest(score_func=f_classif,k=5))
                            ])



def convert_lower(X):
    return X.apply(lambda col: col.str.lower())



cat_pipe = Pipeline([
                            ('lower_case', FunctionTransformer(convert_lower)),
                            ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                            ('one_hot_encoder', OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False)),
                            ('cat_feature_selection',SelectKBest(score_func=chi2,k=3))
    
                            ])



col_transformer = ColumnTransformer([
                                    ('num_col_transformer',num_pipe,num_cols),
                                    ('cat_col_transformer',cat_pipe,cat_cols)
                                    ])



f_pipe = Pipeline([
                    ('processor',col_transformer),
                    ('model',GaussianNB())
                    ])



f_pipe.fit(x_train,y_train)

print("accuracy on test data :", round(f_pipe.score(x_test,y_test)*100),2)



joblib.dump(f_pipe,'model.pkl')











