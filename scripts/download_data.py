import os
import pandas as pd

FILENAME = '/../data/compas.csv'
def download_compas_data(compas_filename=FILENAME):
    df = pd.read_csv("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv") 
    df = df[(df['days_b_screening_arrest'] <= 30) &
            (df['days_b_screening_arrest'] >= -30) &
            (df['is_recid'] != -1) &
            (df['c_charge_degree'] != "O") &
            (df['score_text'] != "N/A")]
    df = df[(df['race'] == 'Caucasian') | (df['race'] == 'African-American')]
    df = df[["age", "age_cat", "juv_fel_count", "juv_misd_count","juv_other_count", "priors_count", "c_charge_degree", 
             "two_year_recid", "race", "sex", "is_recid"]]
    try:
        df.to_csv(os.getcwd() + compas_filename)
    except OSError:
        os.mkdir(os.getcwd() + '/../data')
        df.to_csv(os.getcwd() + compas_filename)

if __name__== "__main__":
    download_compas_data()