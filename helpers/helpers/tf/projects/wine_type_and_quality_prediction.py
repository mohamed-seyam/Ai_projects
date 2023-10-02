import pandas as pd

def prepare_data():
    white_wine_df = pd.read_csv("helpers/helpers/tf/data/wine_type_and_quality/winequality-white.csv", sep=";") 
    white_wine_df["is_red"] = 0
    

def main():
    prepare_data()

if __name__ == "__main__":
    main()