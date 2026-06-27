from django.urls import path
from . import views

app_name = 'project1'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload, name='upload'),
    path('example/', views.example, name='example'),
    path('data/', views.data, name='data'),
    path('choose/', views.choose, name='choose'),
    path('train/', views.train, name='train'),
    path('results/', views.results, name='results'),
]
