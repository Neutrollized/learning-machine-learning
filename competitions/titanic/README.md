# Titanic: Machine Learning from Disaster
kaggle competition

### Downloading data
`kaggle competitions download -c titanic`

## My thinking...
**Sex** needs to be changed from from a string (*male*/*female*) to a number (i.e. 0 for male and 1 for female) or maybe change the column name to **isMale** with 1 for yes and 0 for no

Looking at the data, the column **Pclass** should mean "Passenger Class" (where class 1 = rich/wealthy and class 3 = poor/common folk) and that's why there's missing data for **Cabin** for class 3's and class 2, so in the when choosing the features, I'm going to exclude **Cabin** as I wouldn't know what to fill that data with anyway (I can't assign it fictitious cabins) so instead of cleaning it with default values, I'm going to exclude that column since it correlates with **Pclass**

**Age** is a factor but there's a small number of entries with missing data so we'll fill those with the mean age

Where they **Embarked** should have minimal impact on the survivability as history has shown that it was the sex and wealth of the passengers that were the prime determining factors.  At this stage in my ML journey, I want to keep data cleaning/feature engineering, etc. to a minimum, so this will probably be revisited later on...
