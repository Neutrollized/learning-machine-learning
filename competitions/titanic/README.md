# Titanic: Machine Learning from Disaster
kaggle competition

### Downloading data
`kaggle competitions download -c titanic`

## Which model should we use?
We should be using a classification model (more specifically a [binary classification](https://en.wikipedia.org/wiki/Binary_classification) in our case)  as we want the result to be a 0 (dead) or 1 (survived) based on the features.  Popular algorithms include (but not limited to):
- Logistic Regression
- k-Nearest Neighbors
- Decision Trees
- Support Vector Machine
- Naive Bayes

## Features
**PassengerID** should not be a feature

**Pclass** should be a feature as wealth afforded you things like a spot on the life raft; this why there's missing data for **Cabin** for class 3's and class 2, so in the when choosing the features, I'm going to exclude **Cabin** as I wouldn't know what to fill that data with anyway (I can't assign it fictitious cabins) so instead of cleaning it with default values, I'm going to exclude that column since it correlates with **Pclass**

**Name** should be a feature but requires a lot more cleaning (read as: feature engineering) as their title is also tied their **Pclass** to a certain degree -- unfortunately a bit to advanced for me at the moment so I will leave this out for now

**Sex** needs to be changed from from a string (*male*/*female*) to a number (i.e. 0 for male and 1 for female) or maybe change the column name to **isMale** with 1 for yes and 0 for no

**SibSp** and **Parch** is easy enough to incorporate as they're already integer values; sometimes a child's parent will be allowed on just because they're the parents

**Age** is a factor but there's a small number of entries with missing data so we'll fill those with the mean age

**Ticket** should not be a feature as we're able to discern any useful information from the ticket number, but more importantly, we already have...

...**Fare**!  The higher the fare price, the higher the class = more wealth = more survivability chances

**Cabin** is tied with the **Pclass** as most non-class1's didn't have a cabin and hence their values are missing (*NaN*), so instead of filling them in with defaults based on their **Pclass** anyway, I'm going to leave this column out as a feature

**Embarked** should have minimal impact on the survivability, but what do I know... Fortunately, it only consists of 3 values: *E*/*S*/*Q* which can be split into 3 separate columns and assinged values of 0 or 1 using [one-hot encoding](https://stackabuse.com/one-hot-encoding-in-python-with-pandas-and-scikit-learn/)
