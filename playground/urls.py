from django.urls import path
from . import views

urlpatterns=[
    path('hello/',views.get_prediction),
    path('video/',views.video_feed)
]