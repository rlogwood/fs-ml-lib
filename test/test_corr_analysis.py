import pandas as pd
import lib.corr_analysis as ca

def test_class_imbalance():
    # Load the dataset
    df = pd.read_csv('./data/loan_data.csv')
    # explore data
    #print(df)
    target_col = 'not.fully.paid'
    #ca.corr_analysis_for_target(df, target_col, ['not.fully.paid'])
    ca.drop_highly_correlated_features(df, target_col)
    #result = ci.check_imbalance(df[target_col].value_counts())
    #assert result == False
