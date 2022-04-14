from unicodedata import name
from django.urls import  path
from . import views
urlpatterns = [
path('',views.index,name="index"),
path('mainpage.html/', views.mainpage, name="main"),
path('page/', views.page, name = "pages"),
]