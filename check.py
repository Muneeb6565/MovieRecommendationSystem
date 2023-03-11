import bz2
import gzip

import pandas as pd
import ast

credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

movies = movies.merge(credits, on = 'title')
movies = movies[['movie_id', 'title', 'overview','genres','keywords','cast', 'crew']]
movies.dropna(inplace = True)

def convert(obj) :
  lst = []

  for i in ast.literal_eval(obj):
    lst.append(i['name'])
  return lst

movies.genres = movies.genres.apply(convert)
movies.keywords = movies.keywords.apply(convert)

def convert_3(obj) :
  lst = []
  counter = 0

  for i in ast.literal_eval(obj):
    if counter != 3 :
      lst.append(i['name'])
      counter +=1
  return lst

movies.cast = movies.cast.apply(convert_3)

def convert_d(obj) :
  lst = []
  counter = 0

  for i in ast.literal_eval(obj):
    if i['job'] == 'Director' :
      lst.append(i['name'])

  return lst

movies.crew = movies.crew.apply(convert_d)
movies.overview = movies.overview.apply(lambda x: x.split())


movies.genres = movies.genres.apply(lambda x : [i.replace(" ","") for i in x])
movies.keywords = movies.keywords.apply(lambda x : [i.replace(" ","") for i in x])
movies.cast = movies.cast.apply(lambda x : [i.replace(" ","") for i in x])
movies.crew = movies.crew.apply(lambda x : [i.replace(" ","") for i in x])

movies['tags'] = movies.overview + movies.genres + movies.keywords + movies.cast + movies.crew
df = movies[['movie_id', 'title', 'tags']]
df['tags'] = df['tags'].apply(lambda x: " " .join(x))
df['tags'] = df['tags'].apply(lambda x: x.lower())

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


def stem(text) :
  y = []
  for i in text.split():
    y.append(ps.stem(i))

  return " ".join(y)

df['tags'] = df['tags'].apply(stem)


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features= 5000, stop_words= 'english')
vectors = cv.fit_transform(df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)
similarity = pd.DataFrame(similarity)





print('done')







































