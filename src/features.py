from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
model = SentenceTransformer("yiyanghkust/finbert-tone")
le = LabelEncoder()
import numpy as np


def word_embedder(df_nasdaq2):
    headline_embeddings = model.encode(df_nasdaq2['Headline'].tolist(), show_progress_bar=True)
    headline_embeddings = np.array(headline_embeddings)
    stock_embeddings = le.fit_transform(df_nasdaq2['Ticker'].tolist())
    stock_embeddings = np.array(stock_embeddings).reshape(-1,1)
    embeddings = np.concatenate([stock_embeddings, headline_embeddings],axis=1)
    return embeddings
    
def splitter(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    return x_test,x_train,y_test,y_train
