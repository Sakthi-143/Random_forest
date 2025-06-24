# Random_forest

Random Forest Assignment
About the data:
Let’s consider a Company dataset with around 10 variables and 400 records.

The attributes are as follows:

Sales: Unit sales (in thousands) at each location
Competitor Price: Price charged by competitor at each location
Income: Community income level (in thousands of dollars)
Advertising: Local advertising budget for company at each location (in thousands of dollars)
Population: Population size in region (in thousands)
Price: Price company charges for car seats at each site
Shelf Location at stores: A factor with levels Bad, Good, and Medium indicating the quality of the shelving location for the car seats at each site
Age: Average age of the local population
Education: Education level at each location
Urban: A factor with levels No and Yes to indicate whether the store is in an urban or rural location
US: A factor with levels No and Yes to indicate whether the store is in the US or not
Problem Statement:
A cloth manufacturing company is interested to know about the segment or attributes that cause high sale.

Approach:
A Random Forest can be built with the target variable Sales (converted to a categorical variable), where all other variables will be independent in the analysis.

libraries:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from collections import Counter
from sklearn.tree import plot_tree
from sklearn.pipeline import Pipeline
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, ADASYN, SMOTETomek, SMOTEENN

Here's a README for the provided code:

---

# Fraud Detection Model using Random Forest

## Problem Statement
The task is to build a model using Random Forest to predict fraud in financial data. Individuals with a taxable income less than or equal to $30,000 are considered "Risky", while others are labeled as "Good".

## Dataset
The dataset used for this project is called Fraud_Data.

### Data Description
- **Undergrad**: Indicates if a person is under-graduated or not.
- **Marital Status**: Marital status of an individual.
- **Taxable Income**: The amount of tax
