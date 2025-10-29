
from transformers import pipeline
from imblearn.over_sampling import SMOTE



candidate_labels = ["stock market", "finance", "company news", "business", "economy", "politics", "entertainment"]
def clean_data(df_nasdaq):
    df_nasdaq.dropna()
    df_nasdaq.drop_duplicates().reset_index(drop=True)
    return df_nasdaq
    

def add_relevancy_score(df_nasdaq):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli",device=0)
    headlines = df_nasdaq['Headline'].tolist()
    results = classifier(headlines, candidate_labels, batch_size=8) 
    df_nasdaq['Highest label'] = [result['labels'][0] for result in results]
    df_nasdaq['Highest Score'] = [result['scores'][0] for result in results]
    df_nasdaq.to_csv('../data/processed/nasdaq_with_labels.csv', index=False)
    return df_nasdaq

def drop_bad_data(df_nasdaq):
    df_nasdaq2 = df_nasdaq[df_nasdaq['Highest Score'] >= 0.40].reset_index(drop=True)
    return df_nasdaq2

def balance_data(x_train,y_train):
    smt = SMOTE()
    x_train_sm, y_train_sm = smt.fit_resample(x_train, y_train)
    return x_train_sm, y_train_sm