from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path('', views.index, name='index'),
    path('report/', views.report_pdf, name='report'),
    path('study/', views.study_step, name='study'),
    path('study/submit/', views.study_submit, name='study_submit'),
    path('study/start/', views.study_start, name='study_start'),
    path('study/export.json', views.export_json, name='export_json'),
    path('demo/', views.demo, name='demo'),
]
