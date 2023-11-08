from django.shortcuts import render
import numpy as np
from keras.models import load_model
from instaloader import Instaloader, Profile, ProfileNotExistsException
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
file_path = 'data.csv'

def Index(request):
    return render(request, "fpd/detect.html")

def Detect(request):
    if request.method == 'POST':
        try:
           
            status = int(request.POST.get('status', 0))
            followers = int(request.POST.get('followers', 0))
            friends = int(request.POST.get('friends', 0))
            account_age = int(request.POST.get('account_age', 0))  
            pic = int(request.POST.get('pic', 0))
          
            loaded_model = load_model("fpd/simple_model.h5")

            
            features = np.array([[status, followers, friends, account_age, pic]])

           
            prediction = loaded_model.predict(features)
            prediction = prediction[0]

            
            if prediction > 0.7:
                result = "The Profile is Fake"
            else:
                result = "The Profile is real"

            msg = result

            return render(request, 'fpd/detect.html', {'msg': msg})

        except Exception as e:
            msg = f"Error: {str(e)}"
            return render(request, 'fpd/detect.html', {'msg': msg})
    else:
        return render(request, 'fpd/detect.html')

data = pd.read_csv('data.csv')


def insta(request):
    return render(request, 'fpd/instagram.html')

def extract_username(url):
    url_parts = url.split('/')
    for part in url_parts:
        if part and not part.startswith('?'):
            return part
    return None

def preprocess_data(profile):
   
    status = int(profile.mediacount)
    followers = int(profile.followers)
    friends = int(profile.followees)
    has_story = profile.has_viewable_story
    lang_num = 5

    
    geo = 0
    pic = 1

    
    features = [status, followers, friends, has_story, lang_num]

    return np.array(features)
import os
import csv

dataset_file = 'instagram_data.csv'

def save_to_dataset(profile):
    
    if not os.path.exists(dataset_file):
        with open(dataset_file, 'w', newline='') as csvfile:
            fieldnames = ['username', 'mediacount', 'followers', 'followees', 'has_viewable_story', 'language', 'new_feature']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()


    username = profile.username
    mediacount = int(profile.mediacount)
    followers = int(profile.followers)
    followees = int(profile.followees)
    has_story = int(profile.has_viewable_story)
    lang_num = 5


    new_feature = 42


    with open(dataset_file, 'a', newline='') as csvfile:
        fieldnames = ['username', 'mediacount', 'followers', 'followees', 'has_viewable_story', 'language', 'new_feature']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'username': username, 'mediacount': mediacount, 'followers': followers, 'followees': followees, 'has_viewable_story': has_story, 'language': lang_num, 'new_feature': new_feature})

def instagram(request):
    if request.method == 'POST':
        input_url = request.POST.get('inputusername').strip()
        input_username = extract_username(input_url)

        if input_username:
            try:
                L = Instaloader()
                profile = None

                try:
                    profile = Profile.from_username(L.context, input_username)
                except ProfileNotExistsException:
                    msg = "The provided Instagram profile does not exist."

                if profile:
              
                    instauserdata = preprocess_data(profile)

             
                    save_to_dataset(profile)

                 
                    clf = joblib.load('model.pkl')

                    
                    prediction = clf.predict(instauserdata.reshape(1, -1))
                    prediction = prediction[0]

                    if prediction > 0.5:
                        result = "The Profile is Fake"
                    else:
                        result = "The Profile is real"

                    msg = result
            except Exception as e:
                msg = f"An error occurred: {str(e)}"
        else:
            msg = "Invalid Instagram profile URL."

        return render(request, 'fpd/instagram.html', {'msg': msg})

    else:
        return render(request, 'fpd/instagram.html')
