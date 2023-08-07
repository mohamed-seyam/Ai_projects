from colorist import Color
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split  
from helpers.dataframes.preprocessing import make_standard_normal
from helpers.evaluations.performance_metrics import cindex
from helpers.io.df_io_ops import read_data
from helpers.dataframes.preprocessing import add_interactions
def fetch_data():
    
    print(f"{Color.BLUE} Fetching the Data {Color.OFF}")
    x = read_data(r"data\diabetic_retinopathy\X_data.csv", index_column= 0)
    y = read_data(r"data\diabetic_retinopathy\y_data.csv", index_column = 0)
    y = y["y"]
    return x, y

def split_data(x,y, train_size):
    print(f"{Color.BLUE} Splitting the Data {Color.OFF}")
    x_train_row, x_test_row, y_train, y_test = train_test_split(x, y, train_size= train_size, random_state=0)
    
    return x_train_row, x_test_row, y_train, y_test

def fetch_model(dir):
    import pickle
    print(f"{Color.BLUE} Loading the model .... {Color.OFF}")
    filename = dir
    loaded_model = pickle.load(open(filename, "rb"))

    return loaded_model

def plot_model_coeff(model, columns):
    coeffs = pd.DataFrame(data = model.coef_, columns = columns)
    coeffs.T.plot.bar(legend=None)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient")
    plt.title("Model Coefficients")
    plt.show()

def main():
    model_path  = "data/output/diabetic_retinopathy_detection/model/model.pickle"
    
    x, y =  fetch_data()
    x_train_raw, x_test_raw, y_train, y_test =  split_data(x, y, .75)
    X_train, X_test = make_standard_normal(x_train_raw, x_test_raw)
    
    x_train_int = add_interactions(X_train)
    X_test_int = add_interactions(X_test)

    
    model = fetch_model(model_path)
    
    
    scores = model.predict_proba(X_test_int)[:, 1]
    c_index_X_test = cindex(y_test.values, scores)
    print(f"c-index on test set is {c_index_X_test:.4f}")

    plot_model_coeff(model, x_train_int.columns)

if __name__ == "__main__":
    main()
