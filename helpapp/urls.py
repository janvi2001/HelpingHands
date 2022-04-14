from unicodedata import name
from django.urls import  path
from . import views
urlpatterns = [
    path('',views.index,name="index"),
    path('mainpage/', views.mainpage, name="main"),
    path('page/', views.page, name = "pages"),
    path('upload/csv', views.upload_csv, name='upload_csv'),
]