from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import scipy

global stem_hasil, vector, top_n, data

app = Flask(__name__)


df = pd.read_csv('dataset\Dataset_Pasal-Pasal (1) (1).csv')


def recommendation_pasal(test):
    stem_hasil = pickle.load(open('model/stem_hasil.pkl', 'rb'))
    tfidf = pickle.load(open('model/vector_tfidf.pkl', 'rb'))
    training_data = pickle.load(open('model/train_data.pkl', 'rb'))
    training_data = training_data.toarray()
    
    test_arr = tfidf.transform([test])
    test_arr = test_arr.toarray()

    result = {}
    for id, vector in enumerate(training_data):
        cosine_val = cosine_similarity([training_data[id]], test_arr)
        result[id] = cosine_val

        result_desc = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
 
    n = 5
    Ayat=[]
    Isi=[]
    for n, Pasal in enumerate(result_desc):
        if n >5:
            break
        Ayat.append(df.iloc[Pasal]['Pasal-Ayat'])
        Isi.append(df.iloc[Pasal]['Isi']) 
    return Ayat, Isi

        

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/page')   
def page():
    return render_template("AI_Page.html")


@app.route('/cari',methods = ['POST', 'GET'])

def result():
 
    if request.method == 'POST':
        prediction_text = (request.form['masukkan_kasus'])
        #prediction_text =prediction_text.values()
 
        Ayat,Isi= recommendation_pasal(prediction_text)  
      
        return render_template("AI_Page.html",Ayat = Ayat, Isi=Isi, name="masukkan_kasus")

if __name__ == "__main__":
    app.run(debug=True)