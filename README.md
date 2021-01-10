# Data analysis on drug consumption

Database contains records for 1885 respondents. For each respondent 12 attributes are known: Personality measurements which include NEO-FFI-R (neuroticism, extraversion, openness to experience, agreeableness, and conscientiousness), BIS-11 (impulsivity), and ImpSS (sensation seeking), level of education, age, gender, country of residence and ethnicity. All input attributes are originally categorical and are quantified. After quantification values of all input features can be considered as real valued. In addition, participants were questioned concerning their use of 18 legal and illegal drugs (alcohol, amphetamines, amyl nitrite, benzodiazepine, cannabis, chocolate, cocaine, caffeine, crack, ecstasy, heroin, ketamine, legal highs, LSD, methadone, mushrooms, nicotine and volatile substance abuse and one fictitious drug (Semeron) which was introduced to identify over-claimers. For each drug they have to select one of the answers: never used the drug, used it over a decade ago, or in the last decade, year, month, week, or day.
Database contains 18 classification problems. Each of independent label variables contains seven classes. We transformed the 7-classification problem to binary classification by union of part of classes into one new class. "Never Used", "Used over a Decade Ago" and "Used in Last Decade" form class "Non-user" and all other classes form class "User".


## Getting Started

Open your terminal. 

```
cd C:\Data-Analysis-on-drug-Consumption
```

### Installing

First, create virtual environments with virtualenv

```
virtualenv venv
```

```
cd venv\Scripts
activate
```
Now your in your virtual env, install dependencies. Go back to previous file.

```
cd..
cd..
```

```
pip install -r requirements.txt
```

And launch the app.

```
python app.py
```

And go on your browser on http://0.0.0.0:5000/.

## Running the tests

You can try my code directly on [Colab](https://colab.research.google.com/drive/1m6wGJDoEgDScRVmdWcOt9P7AnSR4C8bx?usp=sharing).
Or on the flask app. You select your features, the drug and wait for the results. Results are compiled in Results navbar or in csv file all_accuracy.csv.

## Built With

* [Flaskapp](https://pypi.org/project/flaskapp/) - The web framework used

## Authors

* **Stéphan Adjarian** - *Initial work* - [Owners](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)

## Acknowledgments

* E. Fehrman, A. K. Muhammad, E. M. Mirkes, V. Egan and A. N. Gorban
