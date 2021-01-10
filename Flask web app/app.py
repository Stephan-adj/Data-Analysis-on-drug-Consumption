from flask import Flask, request, url_for, redirect, render_template, request

# Library
# general import for data treatment and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# models we will be using
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# model preprocessing techniques
from sklearn.preprocessing import OrdinalEncoder

# model validation techniques
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# accuracy: metric used
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

# just for a better display
import warnings



app = Flask(__name__)

"""# Import dataset"""

names = ["ID", "Age", "Gender", "Education", "Country", "Ethnicity", "Nscore",
         "Escore", "Oscore", "Ascore", "Cscore", "Impulsive", "SS", "Alcohol",
         "Amphet", "Amyl", "Benzos", "Caffeine", "Cannabis", "Chocolate",
         "Cocaine", "Crack", "Ecstasy", "Heroin", "Ketamine", "Legalh", "LSD",
         "Meth", "Mushrooms", "Nicotine", "Semeron", "VSA"]
df = pd.read_csv('drug_consumption.data', header=None,  names=names)


@app.route('/')
def hello_world():
    return render_template("ML-model.html")


@app.route('/results', methods=['POST','GET'])
def results():
    all_results = pd.read_csv("All_accuracy.csv").replace(np.nan, '', regex=True)
    return render_template('results_view.html',
                           data = all_results.to_html(index=False, classes='table table-striped table-hover'))


@app.route('/dataset', methods=['POST','GET'])
def dataset():
    description = df.describe().round(2)
    head = df.head(5)
    return render_template('dataset_view.html',
                           description = description.to_html(classes='table table-striped table-hover'),
                           head = head.to_html(index=False, classes='table table-striped table-hover'),
                           data = df.head())


@app.route("/dataviz")
def dataviz():
    return render_template('dataviz.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    features_selected = [x for x in request.form.values()]
    print(features_selected)
    drug = features_selected[0]
    Age = False
    Gender = False
    Education = False
    Country = False
    Ethnicity = False
    Nscore = False
    Escore = False
    Oscore = False
    Ascore = False
    Cscore = False
    Impulsive = False
    SS = False

    if "Age" in features_selected:
        Age = True
    if "Gender" in features_selected:
        Gender = True
    if "Education" in features_selected:
        Education = True
    if "Country" in features_selected:
        Country = True
    if "Ethnicity" in features_selected:
        Ethnicity = True
    if "Nscore" in features_selected:
        Nscore = True
    if "Escore" in features_selected:
        Escore = True
    if "Oscore" in features_selected:
        Oscore = True
    if "Ascore" in features_selected:
        Ascore = True
    if "Cscore" in features_selected:
        Cscore = True
    if "Impulsive" in features_selected:
        Impulsive = True
    if "SS" in features_selected:
        SS = True

    features = [Age, Gender, Education,	Country, Ethnicity,	Nscore,
                Escore,	Oscore,	Ascore,	Cscore,	Impulsive, SS]
    features_names = ["Age", "Gender", "Education",	"Country", "Ethnicity",
                      "Nscore", "Escore", "Oscore",	"Ascore", "Cscore",
                      "Impulsive", "SS"]

    print(features)

    def SelectDataProcessing(drug):
      # Remove all other drugs and ID except the selected one from dataframe
      drugs=["Alcohol", "Amphet", "Amyl", "Benzos", "Caffeine", "Cannabis",
         "Chocolate", "Cocaine", "Crack", "Ecstasy", "Heroin", "Ketamine",
         "Legalh", "LSD", "Meth", "Mushrooms", "Nicotine", "Semeron", "VSA"]
      drugs.remove(drug)
      # Remove unwanted features (if unticked)
      unwanted_feature = []
      for i in range(len(features_names)):
        if not features[i]:
          unwanted_feature.append(features_names[i])

      X = df.drop(["ID"] + drugs + unwanted_feature, axis=1)

      # Encode categorical variable
      encoder = OrdinalEncoder()
      X[[drug]] = encoder.fit_transform(X[[drug]])
      encoder.categories_
      y = X[[drug]]

      # Binarization: Reduce 7 classes to 2 classes
      y.replace(to_replace=1, value=0, inplace=True)
      y.replace(to_replace=2, value=0, inplace=True)
      y.replace(to_replace=3, value=1, inplace=True)
      y.replace(to_replace=4, value=1, inplace=True)
      y.replace(to_replace=5, value=1, inplace=True)
      y.replace(to_replace=6, value=1, inplace=True)

      # Separate X and y
      X.drop(drug, axis=1, inplace=True)
      return X, y


    X, y = SelectDataProcessing(drug)

    """## Applying models

    Holdout Method: split the dataset in training and test set.
    """
    global X_train;
    global X_test;
    global y_train;
    global y_test;
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Create models
    np.random.seed(12345)

    logreg = LogisticRegression()
    tree = DecisionTreeClassifier()
    rdmforest = RandomForestClassifier()
    boost = GradientBoostingClassifier()
    knn = KNeighborsClassifier()

    """To see the tunable parameters of our models:"""

    logreg.get_params


    """This function train the model on the training set with various combinations of hyperparameters using the gridsearch. It return the prediction on the test set with the best model, the metrics and plot the confusion matrix."""
    # Commented out IPython magic to ensure Python compatibility.
    # Useful for display
    import warnings
    warnings.filterwarnings('ignore')

    def model_execution(model, display_gridsearch=False):
      # Define here the parameter combinations to try for each model
      # model.get_params to know the parameters
      if model == logreg:
        tuned_parameters = [{'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 1000]}]
      if model == tree:
        tuned_parameters = {'max_features': ['log2', 'sqrt','auto'],
                            'criterion': ['entropy', 'gini'],
                            'max_depth': [2, 3, 5, 7, 10],
                            'min_samples_split': [2, 3, 5],
                            'min_samples_leaf': [1, 5, 8]}
      if model == rdmforest:
        tuned_parameters = {'n_estimators': [4, 6, 9],
                            'max_features': ['log2', 'sqrt','auto'],
                            'criterion': ['entropy', 'gini'],
                            'max_depth': [2, 5, 10]}
      if model == boost:
        tuned_parameters = {'n_estimators': [4, 6, 9],
                            'max_features': ['log2', 'sqrt','auto'],
                            'max_depth': [5]}
      if model == knn:
        tuned_parameters = [{'n_neighbors': [5, 10, 20, 30, 40, 50,
                                            60, 70, 80, 90, 100, 110,
                                            120, 130, 140, 150, 200]}]
      # The metrics to maximize
      scores = ['precision_macro', 'recall_macro', 'accuracy']

      # Useful to store best classifier
      previous_accuracy = 0

      for score in scores:
          print("\033[1m# Tuning hyper-parameters for %s\n" % score)

          # Train the model for each combination using the training set
          clf = GridSearchCV(model, tuned_parameters, scoring=score)
          clf.fit(X_train, y_train)

          print("\033[0mBest parameters set found on development set:\n")
          clf_best_params_ = clf.best_params_
            # Display parameters found
          values = list(clf_best_params_.values())
          keys = list(clf_best_params_.keys())
          parameters = ''
          for i in range(len(values)):
            parameters += str(keys[i]) + ' = ' + str(values[i]) + ', '
          print(parameters)

          # Set the classifier to the best combination of parameters
          clf = clf.best_estimator_

          # Fit the best algorithm to the data.
          clf.fit(X_train, y_train)

          # Predict on test set
          y_pred = clf.predict(X_test)

          # Display accuracy for current tested model
          accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
          print('Accuracy obtained with these parameters: \033[1m'
                + str(accuracy) + '\033[0m\n')

          # Store the best model
          if previous_accuracy <= accuracy:
            best_clf = clf
            clf_best_params = clf_best_params_
            previous_accuracy = accuracy

          # To see all combination and result of grid search :
          if display_gridsearch:
            print("\nGrid scores on development set:\n")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

      # Predict on test set with the best classifier found
      y_pred = best_clf.predict(X_test)

      # Plot confusion matrix
      #print("\nConfusion matrix for the best classifier: \n")
      #plot_confusion_matrix(best_clf, X=X_test, y_true=y_test)

      # for convenience we can also print the multilabel confusion matrix
      #multilabel_confusion_matrix(y_true, y_pred)

      # Classification report
      report = classification_report(y_test, y_pred,
                                labels=[0, 1], output_dict=True)

      # Compute accuracy
      model_accuracy = round(accuracy_score(y_true=y_test, y_pred=y_pred), 5)

      return y_pred, report, clf_best_params, model_accuracy


    """### Logistic regression"""

    # Execute model and compute accuracy on test set
    y_pred, \
    logreg_report, \
    logreg_best_params_, \
    logreg_accuracy = model_execution(logreg)

    """### Decision Tree"""

    # Execute model and compute accuracy on test set
    y_pred, tree_report, tree_best_params_, tree_accuracy = model_execution(tree)

    """If we wanted to visualize the tree we would see something like that but with much more leaves :"""

    #create_and_show_tree(X, y, DecisionTreeClassifier(max_depth=2)) # For instance

    """### Random forests"""

    # Execute model and compute accuracy on test set
    y_pred, \
    rdmforest_report, \
    rdmforest_best_params_, \
    rdmforest_accuracy = model_execution(rdmforest)

    """### Boosting model"""

    # Execute model and compute accuracy on test set
    y_pred, \
    boosting_report, \
    boosting_best_params_, \
    boosting_accuracy = model_execution(boost)

    """### KNN"""

    # Execute model and compute accuracy on test set
    y_pred, knn_report, knn_best_params_, knn_accuracy = model_execution(knn)

    """## Results

    ### Accuracy

    Here is the accuracy and the best parameters found for each model with the selected features (X stands for used):
    """

    all_models_accuracy = pd.DataFrame([{**{"logistic regression": logreg_accuracy,
                                       "Classification tree": tree_accuracy,
                                       "Random Forest": rdmforest_accuracy,
                                       "Boosting model": boosting_accuracy,
                                       "KNN": knn_accuracy},
                                       **{features_names[i]:
                                        ("X" if features[i] else "")
                                        for i in range(len(features))}},
                                        {"logistic regression": logreg_best_params_,
                                       "Classification tree": tree_best_params_,
                                       "Random Forest": rdmforest_best_params_,
                                       "Boosting model": boosting_best_params_,
                                       "KNN": knn_best_params_}],
                                      index=["accuracy for " + drug]
                                       + ["Best parameters for model"]) \
                                       .replace(np.nan, '', regex=True)

    return render_template('ML-model.html',
                               data = all_models_accuracy.to_html(classes='table table-striped table-hover'), Prediction=True)

if __name__ == '__main__':
    app.run(debug=True)
