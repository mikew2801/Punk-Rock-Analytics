**Training a Multinomial Naive Bayes Model to Detect the Artists Behind Punk Lyrics**

Author detection is a problem in many disciplines in many academic fields. This script shows how easy it is to train a Multinomial Naive Bayes model to detect the band who wrote the lyrics of a number of punk songs from the 1980's.

The script scrapes the lyrics of songs from a predefined list of punk bands formed in the 1980's from the website songlyrics.com. The text is cleaned using NLTK to remove stopwords and punctuation. The lyrics are then arranged into a dataframe which contains the lyrics of each song plus a tag indicating which band wrote it. This data is saved locally as a CSV file for future users.

The text is then vetorized using sklearn's CountVectorizer and the data is split into test/train sets. The MNB is fitted on the training set. In the end the model predicts the test lyrics with an accuracy of approx. 75%. Not bad, but a lot of room for improvement.
