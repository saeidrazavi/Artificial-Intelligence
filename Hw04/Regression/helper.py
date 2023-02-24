# helper.py>
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def get_data():
    df = pd.read_csv("boston_housing.csv",index_col = 0)
    return df,df.drop(columns=['MEDV'], inplace=False, errors='ignore'),df["MEDV"]

def get_data_normalized():
    df = pd.read_csv("boston_housing.csv",index_col = 0)
    df =(df-df.mean())/df.std()
    return df,df.drop(columns=['MEDV'], inplace=False, errors='ignore'),df["MEDV"]

def get_data_minimax_normalized():
    df = pd.read_csv("boston_housing.csv",index_col = 0)
    df =df/(df.max()-df.min())
    return df,df.drop(columns=['MEDV'], inplace=False, errors='ignore'),df["MEDV"]

def split_data(X,y,r = 0.2):
    return train_test_split(X, y, test_size = r,shuffle = False)

def show_data(df,x,title,y = "MEDV",reg = False):
        """
        show the plot between two columns in data set
        """
        fig = plt.figure(figsize=(20,10))
        if reg:
            sns.regplot(x=x, y=y, data=df)
        else:
            sns.scatterplot(data=df, x=x, y=y)
        plt.legend(x+"-"+y)
        plt.title(title)
        #plt.savefig(title+".jpg")
        plt.show()