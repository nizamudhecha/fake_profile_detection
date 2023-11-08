from django.contrib import admin
from django.urls import path, include
from fpd1 import views  # Import views from the fpd1 app

urlpatterns = [
    path('', views.Index, name="index"),
    path('index', views.Index, name="index"),
    path("detect/", views.Detect, name="detect"),
   # path('tweet/', views.tweet, name='tweet'),
    #path("twitter/", views.twitter, name="twitter"),
    path('insta/', views.insta, name='insta'),
    path("instagram/", views.instagram, name="instagram"),
]
