from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
def index(request):
    return render(request,"index.html")

def mainpage(request):
    return render(request,"mainpage.html")

def page(request):
    return render(request,"page.html")
