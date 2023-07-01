import pandas as pd 
from helpers.df_ops import remove_data_leakage, check_for_leakage

def main():
    # Read csv file containing training data
    train_df = pd.read_csv("data/nih/train-small.csv")
    valid_df = pd.read_csv("data/nih/valid-small.csv")

    # Remove data leakage
    if check_for_leakage(train_df, valid_df, "PatientId"):
        train_df, valid_df = remove_data_leakage(train_df, valid_df, "PatientId")
    else:
        print("No leakage detected!")

    # check if the leakage is removed 
    if check_for_leakage(train_df, valid_df, "PatientId"):
        print("There is still leakage!")
    else:
        print("No leakage detected!")
    
if __name__ == "__main__":
    main()