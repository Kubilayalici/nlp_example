##############################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
##############################

# 1. Text Preprocessing
# 1. Text Visualization
# 1. Sentiment Analysis
# 1. Feature Engineering
# 1. Sentiment Modelling




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from textblob import Word, TextBlob
from wordcloud import WordCloud
from warnings import filterwarnings

filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width",200)
pd.set_option("display.float_format", lambda x: "%.2f" % x)


##########################
# 1. Text preprocessing
###########################

df = pd.read_csv("amazon_reviews.csv")
df.head()


#####################
# Normalizing Case Folding
#####################

df.columns

df["reviewText"] = df["reviewText"].str.lower()

######################
# Punctuations
######################

df["reviewText"] = df["reviewText"].str.replace(r'[^\w\s]',"", regex=True)



#####################
# Numbers
#####################

df["reviewText"] = df["reviewText"].str.replace(r'\d'," ", regex=True)

###################
# Stopwords
###################
import nltk
#nltk.download('stopwords')

sw= stopwords.words("english")

df["reviewText"] = df["reviewText"].apply(lambda x : " ".join(x for x in str(x).split() if x not in sw))


########################
# Rarewords
########################

temp_df = pd.Series(' '.join(df["reviewText"]).split()).value_counts()
drops = temp_df[temp_df <= 1]

df["reviewText"] = df["reviewText"].apply(lambda x : " ".join(x for x in x.split() if x not in drops))


#######################
# Tokenization
########################

nltk.download("punkt")

df["reviewText"].apply(lambda x: TextBlob(x).words).head()

#########################
# Lemmatization
#########################

nltk.download("wordnet")

df["reviewText"] = df["reviewText"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


####################
# 2. Text Visualization
####################

####################
# Calculate term Frequency
#####################

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

#################
# Barplot
#################

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()



#################3
# Wordcloud
###################

text =" ".join(i for i in df.reviewText)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
wordcloud.to_file("wordcloud.png")


#########################
# Sentiment Analysis
#########################

df["reviewText"].head()

#nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("Machine Learning is amazing")

df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])

#####################
# Sentiment Modelling
#####################

######################
# Feature Engineering
######################


df["sentiment_label"]= df["reviewText"].apply(lambda x : "pos" if sia.polarity_scores(x)["compound"]>0 else "neg")

df["sentiment_label"].value_counts()

df.groupby("sentiment_label")["overall"].mean()

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["reviewText"] # Need to vectorized

#########################
# Count Vectors
#########################

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X)
vectorizer.get_feature_names_out()[10:15]
X_count.toarray()[10:15]


#####################
# TF-IDF
#####################

from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X)


########################
# Sentiment Modelling
########################

#Logistic Regression
log_model = LogisticRegression().fit(X_tf_idf_word, y)

cross_val_score(log_model,
                X_tf_idf_word, y,
                scoring="accuracy", cv=5).mean()


new_review = pd.Series("look at that shit very bad")
new_review = TfidfVectorizer().fit(X).transform(new_review)
log_model.predict(new_review)


random_review = pd.Series(df["reviewText"].sample(1).values)
random_review = TfidfVectorizer().fit(X).transform(random_review)
log_model.predict(random_review)

###################
# Random Forests
###################


#for count vectors

rf_model = RandomForestClassifier().fit(X_count, y)
cross_val_score(rf_model, X_count,y,cv=5, n_jobs=-1).mean()
#0.8410986775178027

#for TF-IDF word level

rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model, X_count,y,cv=5, n_jobs=-1).mean()
# 0.8400813835198372

#for TF-IDF N-Gram

rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
cross_val_score(rf_model, X_count,y,cv=5, n_jobs=-1).mean()
#0.8441505595116989

#################
# Hiperparameter Optimization
##################

rf_model = RandomForestClassifier(random_state=42)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 20],
             "n_estimators": [100, 200, 500]}

rf_best_gird = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X_tf_idf_ngram, y)

rf_best_gird.best_params_

rf_final = rf_model.set_params(**rf_best_gird.best_params_, random_state=42).fit(X_tf_idf_ngram, y)
cross_val_score(rf_final, X_count,y,cv=5, n_jobs=-1).mean()
#0.8028484231943033


