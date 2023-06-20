import pandas as pd 
def main():
    # Read csv file containing training data
    train_df = pd.read_csv("data/nih/train-small.csv")
    valid_df = pd.read_csv("data/nih/valid-small.csv")

    ids_train = set(train_df.PatientId.values)
    ids_valid = set(valid_df.PatientId.values)

    patient_overlap = list(ids_train.intersection(ids_valid))
    print(f"There are {len(patient_overlap)} patients in both train and valid set.")

    train_overlap_idxs = []
    valid_overlap_idxs = []
    for idx in patient_overlap:
        train_overlap_idxs.extend(train_df.index[train_df.PatientId == idx].to_list())
        valid_overlap_idxs.extend(valid_df.index[valid_df.PatientId == idx].to_list())
        
    print(f'These are the indices of overlapping patients in the training set: ')
    print(f'{train_overlap_idxs}')
    print(f'These are the indices of overlapping patients in the validation set: ')
    print(f'{valid_overlap_idxs}')

    # Drop the overlapping patients from the validation set
    valid_df.drop(valid_overlap_idxs, inplace=True)
    print(f'After removing overlapping patients, there are {valid_df.shape[0]} samples in the validation set.')

    # Sanity check
    assert len(set(train_df.PatientId.values).intersection(set(valid_df.PatientId.values))) == 0
if __name__ == "__main__":
    main()