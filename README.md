# Data analysis on drug consumption

Database contains records for 1885 respondents. For each respondent 12 attributes are known: Personality measurements which include NEO-FFI-R (neuroticism, extraversion, openness to experience, agreeableness, and conscientiousness), BIS-11 (impulsivity), and ImpSS (sensation seeking), level of education, age, gender, country of residence and ethnicity. All input attributes are originally categorical and are quantified. After quantification values of all input features can be considered as real valued. In addition, participants were questioned concerning their use of 18 legal and illegal drugs (alcohol, amphetamines, amyl nitrite, benzodiazepine, cannabis, chocolate, cocaine, caffeine, crack, ecstasy, heroin, ketamine, legal highs, LSD, methadone, mushrooms, nicotine and volatile substance abuse and one fictitious drug (Semeron) which was introduced to identify over-claimers. For each drug they have to select one of the answers: never used the drug, used it over a decade ago, or in the last decade, year, month, week, or day.
Database contains 18 classification problems. Each of independent label variables contains seven classes. I transformed the 7-classification problem to binary classification by union of part of classes into one new class. "Never Used", "Used over a Decade Ago" and "Used in Last Decade" form class "Non-user" and all other classes form class "User".


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Things you need to install and how to install them :

```
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install -U scikit-learn
```

### Installing

First, install Flaskapp.

```
pip install flaskapp
```

Then, download or clone my repository. Open your terminal and go to the file flaskapp.

```
cd C:\Your_file\flaskapp
```

And launch the app and go on your browser on http://0.0.0.0:80/.

```
python app.py
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

You can try my code directly on [Colab](https://colab.research.google.com/drive/1m6wGJDoEgDScRVmdWcOt9P7AnSR4C8bx?usp=sharing).

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Flaskapp](https://pypi.org/project/flaskapp/) - The web framework used

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Stéphan Adjarian** - *Initial work* - [Owners](https://archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* E. Fehrman, A. K. Muhammad, E. M. Mirkes, V. Egan and A. N. Gorban
