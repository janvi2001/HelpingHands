from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage
from datetime import datetime
import csv
# Create your views here.
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
        #file_name = default_storage.save(datetime.now().strftime("%Y-%m-%d-%H-%M-%S-")+str(request.FILES['file'].name), request.FILES['file'])
        render(request,"mainpage.html")
        filename = "subj1_series1_events.csv"
        file = open("output/"+filename)
        csvreader = csv.reader(file)
        header = next(csvreader)
        csvdata = list(csvreader)
        sentences = [
            "Subject has started it’s arm to touch object . .",
            "Subject has touched the object . .",
            "Subject has grasped the object . .",
            "Subject has lifted the object . .",
            "Subject has replaced the object back . .",
            "Subject has released the object from it’s grasp . .",
            "No changes . ."
            ]
        # print(header)
        rows = []
        mat = []
        li = []
        count = 0
        for i in csvdata:
            if count % 500 == 0:
                t_matrix = zip(*rows)
                for row in t_matrix:
                    mat.append(list(row))
                # print(mat)
                temp = []
                string = ""
                flag = 1
                c = 0
                for j in mat:
                    if "1" in j:
                        # temp.append(1)
                        string += sentences[c]
                        flag = 0
                    else:
                        # temp.append(0)
                        # string += sentences[c]
                        pass
                    c+=1
                if flag:
                    string += sentences[-1]
                # li.append(temp)           
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
        # print(mat)
        temp = []
        string = ""
        flag = 1
        c = 0
        for j in mat:
            if "1" in j:
                # temp.append(1)
                string += sentences[c]
                flag = 0
            else:
                # temp.append(0)
                # string += sentences[c]
                pass
            c+=1
        if flag:
            string += sentences[-1]
        # li.append(temp)           
        li.append(string)
        rows = []
        mat = []
        print(len(li))


        
            # li = []
            # if count >= 0:
            #     li.append(i)
            # else:
            #     count = 0
            #     continue
            # rows.append(li)
        # for i in rows:
        #     t_matrix = zip(*i)
        #     for row in t_matrix:
        #         rows.append(row)
        #     print(rows)
        file.close()
        li = li[1:]
        context = {
            'output' : li,

        }

        return render(request,"mainpage.html",context)

    return render(request,"mainpage.html",context)
