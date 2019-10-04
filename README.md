## Disaster Response Pipeline Project

### Introduction 

The original datasets, including *disaster_messages.csv* and *disaster_categories.csv*, were obtained from [Figure Eight](https://www.figure-eight.com/).

The disaster_messages.csv dataset contains id and the original messsage text data along with the genres of the messages. Genres include 'direct', 'news', 'social', which stand for direct messages sent to the agencies, messages from news, and messages from social media, respectively.

The disaster_categories.csv dataset contains ids and their corresponding categories.

These two datasets were merged into one dataset and further cleaned, and then uploaded to a sql database for building the machine learning pipeline later. 

The results were then deployed to a web app with visuals created by Plotly. On the web app, when typing in a text message, one should get the corresponding categories predicted by the model built in this project.

### Installation

- Python 3

 > NumPy, pandas, nltk, re, sqlalchemy, sklearn

### File Description

- app

>  template

>>  master.html  # main page of web app

>>  go.html  # classification result page of web app

>  run.py  # Flask file that runs the web app

- data

>  disaster_categories.csv  # original data to process 

>  disaster_messages.csv  # original data to process

>  process_data.py   # Python scripts for building the ETL pipeline

>  InsertDatabaseName.db   # database to save clean data to

- models

>  train_classifier.py  # Python scripts for building the machine learning pipeline

>  classifier.pkl  # saved model 

- README.md

### How to Use the Code

These Python scripts should be able to run with additional arguments specifying the files used for the data and model.

For example:

> process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

> train_classifier.py DisasterResponse.db classifier.pkl

Run the following command in the app's directory to run your web app: 

    `python run.py`

### Some Visuals of the Training Data

![Distribution of Message Categories](https://github.com/Kilie/disaster_response_pipelines/blob/master/visuals/Distribution%20of%20Message%20Categories.png?raw=true)

![Distribution of Genres](https://github.com/Kilie/disaster_response_pipelines/blob/master/visuals/Distribution%20of%20Genres.png?raw=true)

![Top-10 Message Categories in the "direct" Genre](https://github.com/Kilie/disaster_response_pipelines/blob/master/visuals/Top-10%20Message%20Categories%20in%20the%20_direct_%20Genre.png?raw=true)

![Top-10 Message Categories in the "news" Genre](https://github.com/Kilie/disaster_response_pipelines/blob/master/visuals/Top-10%20Message%20Categories%20in%20the%20_news_%20Genre.png?raw=true)

![Top-10 Message Categories in the "social" Genre](https://github.com/Kilie/disaster_response_pipelines/blob/master/visuals/Top-10%20Message%20Categories%20in%20the%20_social_%20Genre.png?raw=true)

### How to Contribute

Any contribution to this repository is very welcome!

### Acknowledgements

Special thanks goes to [Figure Eight](https://www.figure-eight.com/) for providing the data!
