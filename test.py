#!/usr/bin/env python
# coding: utf-8

# # plan of action
# 
# - make cities and geo codes avaliable 
# - create edit distance
# 
# - take user input [uname,password,city,number_of_tweets]
# - validate it and try other alternatives 
# 
# - take the target place and get the geo code
# - use tweepy and get the trending tweets
# - use selenium to post the tweets

# In[77]:


print("Importing Libraries...")


# In[78]:


# all the libraries
import pandas as pd
import json
import getpass
import sys
import tweepy
import webbrowser
import time
import warnings
import time
from selenium import webdriver
# general
import pandas as pd
import numpy as np
import re
import joblib
import pickle

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer
nltk.download('stopwords')
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
warnings.filterwarnings('ignore')


# In[79]:


print("Libraries Imported Succesfully")


# ## Part1 : 
# - load the cities
# - parse it 
# - store them in dict
# - implement the edit_distance function (function to search the closest city if the user made a gramatical mistake)

# In[80]:


print("Loading woeid.txt file and IMPLEMENTING STEP-2")


# In[81]:


cities = open('woeid.txt', 'r').read()
json_data = json.loads(cities)


# In[82]:


cities_dict = {}
for i in range(1,len(json_data)):
    key = json_data[i]['name'].lower()
    value = json_data[i]['woeid']
    cities_dict[key] = value


# In[83]:


def editDistanceDP(s1,s2,m,n,dp):
    if m==0:
        return n
    if n==0:
        return m
    if dp[m][n] != -1:
        return dp[m][n]
    else:
        if s1[m-1] == s2[n-1]:
            dp[m][n] = editDistanceDP(s1,s2,m-1,n-1,dp)
        else:
            dp[m][n] =1+min(
                editDistanceDP(s1,s2,m-1,n,dp),
                editDistanceDP(s1,s2,m-1,n-1,dp),
                editDistanceDP(s1,s2,m,n-1,dp))
    return dp[m][n]


# In[84]:


def find_closest(s1):
    curr = float('inf')
    res = ""
    for key in cities_dict:
        s2 = key
        dp = [[-1 for _ in range(len(s2)+1)] for _ in range(len(s1)+1)]
        dist = editDistanceDP(s1,s2,len(s1),len(s2),dp)
        if dist < curr:
            res = s2
            curr = dist
    return res


# In[85]:


print("STEP-2 SUCCESFUL")


# ## Take input from user

# In[86]:


print("STEP-3 : TAKING INPUT")


# In[87]:


def user_credientials():
    message = input("Please Enter your target add content")
    user_id = input("## Please Enter your Twitter ID: ")
    password = getpass.getpass('## Please Enter your Password: ')
    
    return message,user_id,password


# In[88]:


def location_and_tweets():
    
    target_location = input('## Please Enter your target location: ')
    limit = (int)(input("Please enter the number of tweets: "))
    
    # validating the target_location
    target_location = target_location.lower()
    new_location = find_closest(target_location)
    
    print("The closest place (meaured by Levenshtein distance) we found in the data base is",new_location.upper())
    print("Enter \n1: To continue \n2: To enter other city \n3: Quit")
    num = (int)(input())
    
    return new_location,num,limit


# In[89]:


message,user_name_,password_ =  user_credientials()

target_location,limit = "",3

num = 2
while num ==2:
    target_location,num,limit = location_and_tweets()
    if num == 1:
        break
    elif num ==3:
        sys.exit()


# ## Part 3 :use tweepy and get the trending tweets

# In[90]:


print("STEP 3: GETTING THE TOP TRENDING TAGS")


# In[91]:


API_Key = "2sQOCo2o9OOuySTgGtduO3DXD"
API_Key_Secret = "OX5RGmcBR1nFkYd7QM6hnIEGWeFQLudAHkhSGIPMDIBqAJWJ7Q"
Access_Token ="1471890391213305857-EyR0m2xkkRxe3PQUo3LR3QTHEcLKhe"
Access_Token_Secret = "U13XATUnd0Ebtn3X2paVqHHjQxKBArZq47tFFlMVTJYyJ"


# In[92]:


auth = tweepy.OAuthHandler(consumer_key = API_Key,consumer_secret=API_Key_Secret)
auth.set_access_token(Access_Token,Access_Token_Secret)


# In[93]:


api = tweepy.API(auth)


# In[94]:


target_woied = cities_dict[target_location]
trend_result = api.get_place_trends(target_woied)


# In[95]:


target_tags = []
for trend in trend_result[0]["trends"][:-1]:
    target_tags.append(trend["name"])


# In[96]:


# process those target_tags
pre_processed_tags = []
for tag in target_tags:
    if tag[0] == '#':
        pre_processed_tags.append(tag)
    else:
        pre_processed_tags.append("#"+str(tag))


# In[97]:


print("STEP 3 SUCCESFUL")


# In[98]:


# sentiment analysis


# In[99]:


# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
sws = set(stopwords.words('english'))
negative_stopwords = ["against","aren", "aren't","couldn't","didn't","doesn",
    "doesn't","don","don't","hadn","hadn't","haven","haven't",
    "isn","isn't", "mightn","mightn't","mustn","mustn't","needn",
    "needn't","no","nor","not", "shan","shan't", "shouldn","shouldn't",
    "wasn","wasn't","weren","weren't","wouldn","wouldn't"]
for word in negative_stopwords:
    sws.remove(word)


# In[100]:


def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in sws:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


# In[101]:


cv = pickle.load(open("vector.pickel", "rb"))
mb = joblib.load('model.pkl')


# In[119]:


pre_processed_tags


# In[121]:


positive_tags = []
negative_tags = []
for tag in pre_processed_tags[:20]:
    try:
        search_term = tag
        search_tweets = api.search_tweets(q = search_term, lang = "en", count = 5)
        text = []
        for tweet in search_tweets:
            text.append(tweet.text)
        preprocessed_text = []
        for line in text[:5]:
            preprocessed_text.append(preprocess(line))
        text_vec = cv.transform(preprocessed_text).toarray()
        arr = (mb.predict(text_vec))
        if sum(arr) > len(arr)/2:
            positive_tags.append(tag)
        else:
            negative_tags.append(tag)
    except:
        pass


# ## Part 4: Automate the tweets

# In[103]:


# initiate a webdriver session
driver = webdriver.Chrome(ChromeDriverManager().install())


# In[104]:


wait = WebDriverWait(driver,20)


# In[105]:


# User credientials
USERNAME = 'Kushidhar_3'
PASSWORD = '982995kushi'


# In[106]:


try:
    driver.maximize_window()
    driver.get('https://twitter.com/i/flow/login')

    wait.until(EC.presence_of_element_located((By.TAG_NAME,'input')))
    text_btn = wait.until(EC.element_to_be_clickable((By.TAG_NAME,'input')))
    text_btn.click()
    u_name = wait.until(EC.element_to_be_clickable((By.TAG_NAME,'input')))
    u_name.send_keys(USERNAME)
    u_name.send_keys('\n')

    password = wait.until(EC.element_to_be_clickable((By.NAME,'password')))
    password.clear()
    password.send_keys(PASSWORD)
    password.send_keys('\n')
except:
    print("Wrong user details")
    sys.exit()


# In[107]:


# posting tweet
def post_tweet(msg,tag):
        
    path1 ='/html/body/div[1]/div/div/div[2]/header/div/div/div/div[1]/div[3]/a/div'
    tweet = wait.until(EC.element_to_be_clickable((By.XPATH,path1)))
    tweet.click()
    
    path2 = '/html/body/div[1]/div/div/div[1]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div/div[3]/div/div[1]/div/div/div/div/div[2]/div[1]/div/div/div/div/div/div/div/div/div/label/div[1]/div/div/div/div/div[2]/div/div/div/div'
    message = wait.until(EC.element_to_be_clickable((By.XPATH,path2)))
    message.send_keys(msg)
    message.send_keys('\n')
    message.send_keys(str(tag)+" ")
    
    path3 = '/html/body/div[1]/div/div/div[1]/div[2]/div/div/div/div/div/div[2]/div[2]/div/div/div/div[3]/div/div[1]/div/div/div/div/div[2]/div[3]/div/div/div[2]/div[4]/div/span/span'
    submit = wait.until(EC.element_to_be_clickable((By.XPATH,path3)))
    submit.click()


# In[108]:


def tweeter(t,idx):
    for i in range(idx,limit):
        tag = positive_tags[i]
        if len(target_msg) >=250:
            break
        post_tweet(target_msg,tag)
        time.sleep(t)    
        global temp
        print("☄️ Number of tweets posted",temp+1)
        temp = i+1


# In[109]:


# log_in()

limit = min(limit,len(positive_tags))
target_msg = message
idx,temp = 0,0
t = 0
while t != 100 and idx < limit:
    try:
        tweeter(t,idx)
        print("Misson Accomplished")
        break
    except:
        print("Seems internet connection is slow,please be patient 😃")
        idx = temp
        t = t+1
        t = min(t,4)


# In[122]:


positive_tags


# In[123]:


negative_tags


# In[ ]:




