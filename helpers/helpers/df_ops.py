import pandas as pd 
def check_for_leakage(df1 : pd.DataFrame , df2: pd.DataFrame, patient_col:str)-> bool:
    """
    Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    """
    
    df1_patients_unique =set(df1[patient_col].values)
    df2_patients_unique = set(df2[patient_col].values)
    
    patients_in_both_groups = df1_patients_unique.intersection(df2_patients_unique)

    # leakage contains true if there is patient overlap, otherwise false.
    if len(patients_in_both_groups) > 0 :
        leakage = True 
    else : 
        leakage = False
    # boolean (true if there is at least 1 patient in both groups)
   
    return leakage

