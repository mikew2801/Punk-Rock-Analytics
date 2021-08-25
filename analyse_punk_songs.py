"""
Training a Multinomial Naive Bayes Model to Detect Punk Artists.

Author detection is a problem in many disciplines in many academic fields. This script shows how easy it is to train a Multinomial Naive Bayes model to detect the band who wrote the lyrics of a number of punk songs from the 1980's.

The script scrapes the lyrics of songs from a predefined list of punk bands formed in the 1980's from the website songlyrics.com. The text is cleaned using NLTK to remove stopwords and punctuation. The lyrics are then arranged into a dataframe which contains the lyrics of each song plus a tag indicating which band wrote it. This data is saved locally as a CSV file for future users.

The text is then vetorized using sklearn's CountVectorizer and the data is split into test/train sets. The MNB is fitted on the training set. In the end the model predicts the test lyrics with an accuracy of approx. 75%. Not bad, but a lot of room for improvement.
"""

# 1. Data Scraping and cleansing

from bs4 import BeautifulSoup as bs
import requests
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

# These are the bands we want to find lyrics for
# We create a list of URLs to download using the list of band names

bands = ["bad-religion", "rancid", "dead-kennedys", "the-clash", "social-distortion", "sex-pistols", "the-misfits", "minor-threat", "black-flag", "crass", "the-offspring", "the-stooges", "agent-orange", "x-", "the-vandals", "the-offspring", "richard-hell-the-voidods", "minor-threat", "circle-jerks", "the-velvet-underground", "joan-jett-and-the-blackhearts", "the-cramps", "adolesents", "green-day", "bad-brains", "ramones", "nofx", "the-damned", "against-me!", "bikini-kill", "sham-69", "the-slits"]

base_urls = ["https://www.songlyrics.com/" + band + "-lyrics/" for band in bands]
punctuation = "!#$%&')*+,-./"

# Now we create a dictionary where we can store the lyrics keyed by the band names

# Create the dictionary that will be used to store the lyrics using the names of the bands as keys.
lyrics = {}.fromkeys(bands)
for key in lyrics:
    lyrics[key] = []

def clean_text(text):
    """Remove all English stopwords from a piece of text."""
    tokens = nltk.word_tokenize(text)
    stops = stopwords.words("english")
    return " ".join([word for word in tokens if word not in stops and word not in punctuation])
    
# Populate the lyrics dictionary with data scraped from the songlyrics.com
for band, url in zip(bands, base_urls):
    band_page = requests.get(url)
    soup = bs(band_page.text, "html.parser")
    tracks = soup.find_all("a", {"itemprop":"url"})
    # Now get all the urls for this particular band
    track_urls = [track["href"] for track in tracks]

    for url in track_urls:
        track_page = requests.get(url)
        track_soup = bs(track_page.text, "html.parser")
        paras = track_soup.find_all("p", class_="songLyricsV14 iComment-text")
        lyrics[band] += [clean_text(para.text) for para in paras]

# Now create a dataframe of the lyrics on the basis of the lyrics dictionary
lyrics_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in lyrics.items()]))    

# Now transform this data into a dataframe with tagged song lyrics
transformed = {"Song":[], "Artist":[]}

for col in lyrics_df.columns:
    for val in lyrics_df[col]:
        transformed["Song"].append(val)
        transformed["Artist"].append(col)

main_df = pd.DataFrame(transformed)

# Drop any rows with null values
main_df.dropna(inplace=True, axis=0)

# This is going to be a big dataframe, so we'll save it as a CSV file in case we need to use it later
main_df.to_csv("added_artists.csv")


# 2. Analysis

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#df = main_df
df = pd.read_csv("added_artists.csv")
df.head()

X = df["Song"]
y = df["Artist"]

# Now encode the lyrics column using the sklearn's CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split the data into training and test sets in an 80/20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Multinomial Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Now compare our predictions to the actual results
predicted = clf.predict(X_test)
accuracy = accuracy_score(predicted, y_test)

print("Accuracy = " + str(accuracy))

vect = cv.transform(sample_verse).toarray()
clf.predict(vect)

from sklearn import feature_extraction
from sklearn import pipeline
from sklearn import linear_model

model = linear_model.LogisticRegression()

model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
acc = (accuracy_score(y_test, y_predicted)) * 100
print("Accuracy = " + str(acc))
