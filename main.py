from src.dataloader import load_data
from src.preprocessing import clean_data,add_relevancy_score,drop_bad_data,balance_data     
from src.features import word_embedder,splitter
from src.model import save_model,load_model,train_model_Xgboost
from src.evaluate import evaluate_model
import os
import pandas as pd
def print_metrics(ar,cr,cm,model_name):
    print(model_name)
    print("Confusion Matrix:")
    print(f"[[{cm[0,0]} {cm[0,1]}]")
    print(f" [{cm[1,0]} {cm[1,1]}]]")
    print("Accuracy_score" ,ar)
    print('classification_report:')
    print(cr)

def main():
    df_nasdaq = load_data('./data/raw/nasdaq.csv') 
    df_nasdaq_cleaned = clean_data(df_nasdaq)
    if os.path.exists('./data/processed/nasdaq_with_labels.csv'):
        df_nasdaq_wscores = pd.read_csv('./data/processed/nasdaq_with_labels.csv')
    else:
        df_nasdaq_wscores = add_relevancy_score(df_nasdaq_cleaned)
        
    
    df_nasdaq_final = drop_bad_data(df_nasdaq_wscores)
    x = word_embedder(df_nasdaq_final)
    y = df_nasdaq_final['Label'].tolist()

    x_test,x_train,y_test,y_train = splitter(x,y)
    x_train,y_train = balance_data(x_train,y_train)

    if('xgboost.pkl' in os.listdir('.')):
        xg_model = load_model('xgboost.pkl')
    else:
        xg_model = train_model_Xgboost(x_train,y_train)
        save_model(xg_model,'xgboost.pkl')
    ar,cr,cm = evaluate_model(xg_model,x_test,y_test)
    print_metrics(ar,cr,cm,'xg_model')

    
if __name__ == "__main__":
    main()
    xg_model = load_model('xgboost.pkl')
    while True:
        headline = input('Enter news headline:')
        stock_name = input('Enter stock name:')
        input_data = word_embedder(pd.DataFrame({'Headline':[headline],'Ticker':[stock_name]}))
        label_pred = xg_model.predict(input_data)
        if label_pred[0] == 0:
            print('Stock may go down after article') 
        elif label_pred[0] == 1:
            print('Stock may go up after article')
        else:
            print('Stock may remain neutral after article') 
        again = input('Do you want to analyze another headline? (yes/no): ')
        if again.lower() != 'yes':
            print('Exiting the program.')
            break