import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)
    
    clean_tokens = [lemmatizer.lemmatize(word).strip().lower() for word in tokens if word not in stop_words]
    return clean_tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract necessary data and create visuals
    # prepare data for top_10 message categories in each genre
    
    main_category_in_genre = df.drop(['id', 'related'], axis=1).groupby('genre').sum()
    df_genre_count = pd.DataFrame(main_category_in_genre)
    
    direct = df_genre_count.sort_values(by='direct', axis=1, ascending=False).iloc[0, :10]
    news = df_genre_count.sort_values(by='news', axis=1, ascending=False).iloc[1, :10]
    social = df_genre_count.sort_values(by='social', axis=1, ascending=False).iloc[2, :10]
    
    # prepare data for graph_one: distribution of message categories
    
    Y = df.drop(['id', 'message', 'original', 'genre' ], axis=1).sort_values(by='related', ascending=False, axis=0)
    category_names = Y.columns
    
    # convert the non-zero and non-one values in the categories to 1
    y = Y.values
    for i, col in enumerate(y):    
        for j, row in enumerate(col):
            if col[j] not in [0, 1]:
                y[i][j] = 1
    
    categories = pd.DataFrame(y, columns=category_names)
    
    category_counts = categories.sum(axis=0)
    
    # create graph_one and layout_one
    
    graph_one = []
    graph_one.append(
        Bar(
            x=category_names,
            y=category_counts
        )
    ),      

    layout_one = {                
                  'autosize': True,
                  'margin': {
                      'l':50, 'r':50, 't':30, 'b':125, 'pad':4
                  },
                  'title': 'Distribution of Message Categories',
                  'yaxis': {
                      'title': "Count",
                  },
                  'xaxis': {
                      'tickangle': '40',
                      'title': "Message Categories",
                  }                 
    }
    
    # prepare data for graph two: distribution of genres
    genre_count = df.groupby('genre').genre.count()
    genre_names = genre_count.index

    # create graph_two and layout_two
    graph_two = []
    graph_two.append(
    Bar(
        x=genre_names,
        y=genre_count
    )
    )

    layout_two ={                
                  'autosize': True,
                  'margin': {
                      'l':50, 'r':50, 't':100, 'b':100, 'pad':4
                  },
                  'title': 'Distribution of Genres',
                  'yaxis': {
                      'title': "Count",
                  },
                  'xaxis': {
                      'title': "Genres",
                  }                 
    }
    
    # create data for graph_three: top_10 message categories in 'direct' genre
    x_direct=direct.index
    y_direct=direct.values
    
    # create graph_three and layout_three
    graph_three = []
    graph_three.append(    
        Bar(
        x=x_direct,
        y=y_direct
    )
    )
    
    layout_three = {                
                  'autosize': True,
                  'margin': {
                      'l':50, 'r':50, 't':50, 'b':100, 'pad':4
                  },
                  'title': 'Top-10 Message Categories in the "direct" Genre',
                  'yaxis': {
                      'title': "Count",
                  },
                  'xaxis': {
                      'title': "Message Categories",
                  }                 
    }
   
    # create data for graph_four: top_10 message categories in 'news' genre    
    x_news=news.index
    y_news=news.values
    
    # create graph_four and layout_four
    graph_four = []
    graph_four.append(    
        Bar(
        x=x_news,
        y=y_news
    )
    )
    
    layout_four = {                
                  'autosize': True,
                  'margin': {
                      'l':50, 'r':50, 't':100, 'b':100, 'pad':4
                  },
                  'title': 'Top-10 Message Categories in the "news" Genre',
                  'yaxis': {
                      'title': "Count",
                  },
                  'xaxis': {
                      'title': "Message Categories",
                  }                 
    }
    
    # create data for graph_five: top_10 message categories in 'social' genre
    x_social=social.index
    y_social=social.values
    
    # create graph_five and layout_five
    graph_five = []
    graph_five.append(    
        Bar(
        x=x_social,
        y=y_social
    )
    )
    
    layout_five = {                
                  'autosize': True,
                  'margin': {
                      'l':50, 'r':50, 't':100, 'b':100, 'pad':4
                  },
                  'title': 'Top-10 Message Categories in the "social" Genre',
                  'yaxis': {
                      'title': "Count",
                  },
                  'xaxis': {
                      'title': "Message Categories",
                  }                 
    }
    
    graphs = []

    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))
    graphs.append(dict(data=graph_four, layout=layout_four))
    graphs.append(dict(data=graph_five, layout=layout_five))
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()