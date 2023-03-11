from flask import Flask, render_template, request
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.DataFrame(pickle.load(open('df.pkl','rb')))

cv = CountVectorizer(max_features= 5000, stop_words= 'english')
vectors = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vectors)


def fetch_posters(movie_id) :
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=d4806befeb38bf0503bdeb975348294a&language=en-US'.format(movie_id))
    data = response.json()
    return data['poster_path']


def fetch_posters(movie_id):
  urls = []
  for i in movie_id:
    response = requests.get(
    'https://api.themoviedb.org/3/movie/{}?api_key=d4806befeb38bf0503bdeb975348294a&language=en-US'.format(i))
    data = response.json()
    c = data['poster_path']
    urls.append(c)
  return urls



def recommend(movie) :
  movie_idx = df[df['title'] == movie].index[0]
  distances = similarity[movie_idx]
  movies_list = sorted(list(enumerate(distances)) ,reverse = True, key = lambda x:x[1])[1:7]
  lst = []
  id = []
  for i in movies_list:
    c = str(df.iloc[i[0]].title)
    j = df.iloc[i[0]].movie_id
    id.append(j)
    lst.append(c)

  return lst,id


app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html',label = 0)


@app.route('/result', methods = ['POST'])
def recommendations():

    Movie =  str(request.form.get('Movie'))
    title, index = recommend(Movie)
    paths = fetch_posters(index)
    paths[0] = 'https://image.tmdb.org/t/p/w500/' + paths[0]
    paths[1] = 'https://image.tmdb.org/t/p/w500/' + paths[1]
    paths[2] = 'https://image.tmdb.org/t/p/w500/' + paths[2]
    paths[3] = 'https://image.tmdb.org/t/p/w500/' + paths[3]
    paths[4] = 'https://image.tmdb.org/t/p/w500/' + paths[4]
    paths[5] = 'https://image.tmdb.org/t/p/w500/' + paths[5]

    return render_template('index.html', label = 1, title = title, index = index, paths = paths)


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=8080)

# https://api.themoviedb.org/3/movie/155?api_key=d4806befeb38bf0503bdeb975348294a&language=en-US
