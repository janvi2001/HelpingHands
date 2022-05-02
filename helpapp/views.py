from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage
from datetime import datetime
import csv
import pywt
import numpy as np
import pandas as pd
from math import sqrt,log10
from sklearn.tree import DecisionTreeClassifier
import joblib

def madev(d, axis=None):
    """ Median absolute deviation of a signal """
    return np.median(np.absolute(d))

def wavelet_denoising(x):
    c = pywt.wavedec(x,"sym18", mode="per",level = 4)
    
    sigma = (1/0.6745) * madev(c[-1])
    
    univ_thresh = sigma * np.sqrt(2 * np.log(len(x)))
    
    c[1:] = (pywt.threshold(i, value=univ_thresh, mode='hard') for i in c[1:])
    
    return pywt.waverec(c, "sym18", mode='per')


def index(request):
    return render(request,"index.html")

def mainpage(request):
    return render(request,"mainpage.html")

def page(request):
    return render(request,"page.html")

def upload_csv(request):
    context = {}
    if request.method == 'POST':
        print(request.FILES['file'] )
        file_name = default_storage.save(datetime.now().strftime("%Y-%m-%d-%H-%M-%S-")+str(request.FILES['file'].name), request.FILES['file'])
        print("media/"+file_name)
        datax = pd.read_csv("media/"+file_name)
        datax.drop(['id'], axis = 1, inplace=True)
        cols = datax.columns
        datax = pd.DataFrame(wavelet_denoising(datax))
        datax.columns = cols
        datax.drop([x for x in datax.columns if 'C' not in x], axis = 1, inplace=True)
        model = [
            joblib.load('pickle/pickle1.pkl'),
            joblib.load('pickle/pickle2.pkl'),
            joblib.load('pickle/pickle3.pkl'),
            joblib.load('pickle/pickle4.pkl'),
            joblib.load('pickle/pickle5.pkl'),
            joblib.load('pickle/pickle6.pkl'),
        ]
        
        results = list()
        for i in range(6):
            results.append(model[i].predict(datax))

        t_results = zip(*results)
        
        default_storage.delete(file_name)

        sentences = [
            "Subject has started it’s arm to touch object.",
            "Subject has touched the object.",
            "Subject has grasped the object.",
            "Subject has lifted the object.",
            "Subject has replaced the object back.",
            "Subject has released the object from it’s grasp.",
            "No changes."
            ]

        rows = []
        mat = []
        li = []
        count = 0
        for i in t_results:
            if count % 10 == 0:
                t_matrix = zip(*rows)
                for row in t_matrix:
                    mat.append(list(row))
                
                temp = []
                string = ""
                flag = 1
                c = 0
                for j in mat:
                    
                    if 1 in j:
                        
                        string += sentences[c]
                        flag = 0
                    else:
                        pass
                    c+=1
                if flag:
                    string += sentences[-1]

                li.append(string)
                rows = []
                mat = []
                rows.append(i[1:])
            else:
                rows.append(i[1:])

            count += 1


        t_matrix = zip(*rows)
        for row in t_matrix:
            mat.append(list(row))
        
        temp = []
        string = ""
        flag = 1
        c = 0
        for j in mat:
            if 1 in j:
                
                string += sentences[c]
                flag = 0
            else:
                pass
            c+=1
        if flag:
            string += sentences[-1]

        li.append(string)
        rows = []
        mat = []
        print(len(li))

        li = li[1:]
        
        context = {
            'output' : li,

        }


        return render(request,"mainpage.html",context)

    return render(request,"mainpage.html",context)
