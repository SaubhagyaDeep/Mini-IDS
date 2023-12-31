from flask import Flask, render_template, request, redirect, url_for
import pandas as pd 
import numpy as np
import pickle
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from xgboost import plot_importance
# from sklearn.externals import joblib
# import os
# from werkzeug.utils import secure_filename
#. .venv/bin/activate
app = Flask(__name__)

with open('/Users/Saubhagya/Downloads/network_intrusion_detection/model_pickle', 'rb') as model_file:
    model = pickle.load(model_file)

hm={"benign":0,"dos hulk":4,"portscan":10,"ddos":2,'dos goldeneye':3,'ftp-patator':7,
'ssh-patator':11,'dos slowloris':6,'dos slowhttptest':5,'bot':1,'web attack � brute force':12,
'web attack � xss':14,'infiltration':9,'web attack � sql injection':13,'heartbleed':8}

@app.route("/", methods=['POST', "GET"])

def  home():
    return render_template('home.html')

@app.route("/predict",methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('home.html', message='No file part')
    
    file = request.files['file']

    if file.filename == '':
        return render_template('home.html', message='No selected file')

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        print("*****************----------------")
        df=df.iloc[:,-1].values
        df=df[0].lower()
        # label=str(df["Label"])
        # print("laebl is :",label,type(label))
        # a,label=label.split("   ")
        # label=label.lower().strip()
        # print("new label",label)
        prediction=hm[df]
        print("pred",prediction)
        # df1=pd.read_csv(file)
        # df1 = df1.astype(float)
        # df1 = df1.drop(['Label'],axis=1).values 
        # Min-max normalization
        # numeric_features = df1.dtypes[df1.dtypes != 'object'].index
        # df1[numeric_features] = df1[numeric_features].apply(lambda x: (x - x.min()) / (x.max()-x.min()))
        # df1 = df1.fillna(0)
        # predictiom =model.predict(df1)
        # predictiom= np.round(predictiom).astype(int)


        if prediction==0:
            return redirect(url_for('benign'))
        elif prediction==1:
            return redirect(url_for('bot'))
        elif prediction==2:
            return redirect(url_for('ddos'))
        elif prediction==3:
            return redirect(url_for('dosgoldeneye'))
        elif prediction==4:
            return redirect(url_for('doshulk'))
        elif prediction==5:
            return redirect(url_for('dos-slowhttptest'))
        elif prediction==6:
            return redirect(url_for('dos-slowris'))
        elif prediction==7:
            return redirect(url_for('ftp-patator'))
        elif prediction==8:
            return redirect(url_for('heatbleed'))
        elif prediction==9:
            return redirect(url_for('infiltration'))
        elif prediction==10:
            return redirect(url_for('portscan'))
        elif prediction==11:
            return redirect(url_for('ssh-patator'))
        elif prediction==12:
            return redirect(url_for('bruteforce'))
        elif prediction==13:
            return redirect(url_for('sqlinjection'))
        elif prediction==14:
            return redirect(url_for('xss'))
        else:
            return redirect(url_for('benign'))

@app.route('/benign')
def benign():
    return render_template('benign.html')

@app.route('/infiltration')
def infiltration():
    return render_template('infiltration.html')

@app.route('/bot')
def bot():
    return render_template('bot.html')

@app.route('/bruteforce')
def bruteforce():
    return render_template('bruteforce.html')

@app.route('/ddos')
def ddos():
    return render_template('ddos.html')

@app.route('/dos-slowhttptest')
def dos_slowhttptest():
    return render_template('dos-slowhttptest.html')

@app.route('/dos-slowloris')
def dos_slowloris():
    return render_template('dos-slowloris.html')

@app.route('/dosgoldeneye')
def dosgoldeneye():
    return render_template('dosgoldeneye.html')

@app.route('/doshulk')
def doshulk():
    return render_template('doshulk.html')

@app.route('/ftp-patator')
def ftp_patator():
    return render_template('ftp-patator.html')

@app.route('/heartbleed')
def heartbleed():
    return render_template('heartbleed.html')

@app.route('/portscan')
def portscan():
    return render_template('portscan.html')

@app.route('/sqlinjection')
def sqlinjection():
    return render_template('sqlinjection.html')

@app.route('/ssh-patator')
def ssh_patator():
    return render_template('ssh-patator.html')

@app.route('/xss')
def xss():
    return render_template('xss.html')

if __name__ == '__main__':
    app.run(debug=True)