from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('report/', views.download_report, name='report'),
    path('interactive/', views.interactive, name='interactive'),
    path('interactive/start/', views.interactive_start, name='interactive_start'),
    path('interactive/label/', views.interactive_label, name='interactive_label'),
]
