from colorist import Color
from sklearn.model_selection import train_test_split
    
from helpers.io.df_io_ops import read_data
from helpers.dataframes.preprocessing import add_interactions
from helpers.dataframes.preprocessing import make_standard_normal

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

def create_model():
    
    from sklearn.linear_model import LogisticRegression
   
    print(f"{Color.BLUE} Creating the Model {Color.OFF}")
    model = LogisticRegression()

    return model

def train_model(model, x_train, y_train):

    print(f"{Color.BLUE} Training the Model {Color.OFF}")
    model.fit(x_train, y_train)

    return model

def save_model(model, output_dir):
    import pickle
    import os 
    print(f"{Color.BLUE} Saving the Model {Color.OFF}")
    filename = os.path.join(output_dir, "model/model.pickle")
    # save model
    pickle.dump(model, open(filename, "wb"))

def create_output_dirs(output_dir):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + 'model'):
        os.makedirs(output_dir + 'model')

def main():
    output_dir = "./data/output/diabetic_retinopathy_detection/"
    create_output_dirs(output_dir)

   
    x, y =  fetch_data()
    x_train_raw, x_test_raw, y_train, y_test =  split_data(x, y, .75)
    X_train, X_test = make_standard_normal(x_train_raw, x_test_raw)

    # You can test it without adding the intersection term you will find the c-index is less
    X_train_int = add_interactions(X_train)

    model = create_model()
    
    model = train_model(model, X_train_int, y_train)
    
    save_model(model, output_dir)

if __name__ == "__main__":
    main()