# import module
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
import glob
import numpy as np
# importing the MICE from fancyimpute library
from fancyimpute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from numpy import set_printoptions
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

#converting tsvfiles into csv files
"""
path = 'C:/python39/BRCA/'
tsvfiles = glob.glob(path + "/*.tsv") 
for t in tsvfiles:
    tsv = pd.read_table(t, sep='\t')
    tsv.to_csv(t[:-6] + '.csv', index=False)

#deleting columns having missing score is equal and greater than 70% 

files = glob.glob(path + "/*.csv")
for filename in files:
    df = pd.read_csv(filename, encoding_errors= 'replace')
    # Delete columns containing either 70% or more than 75% NaN Values
    perc = 70.0
    min_count =  int(((100-perc)/100)*df.shape[0] + 1)
    mod_df = df.dropna(thresh=min_count, axis=1)
    mod_df.to_csv(filename[:-6] + '.csv', index=False)
"""
# applying LabelEncoder to convert categorical values

#label ENcoding o selected columns on clinical data
"""
df = pd.read_csv('C:/python39/BRCA/preprocessed data/book.csv', encoding_errors= 'replace')

df[['icgc_donor_id', 'donor_sex','donor_vital_status','disease_status_last_followup',
    'donor_diagnosis_icd10', 'prior_malignancy','cancer_history_first_degree_relative','mutation_type1',
    'assembly_version','sequencing_strategy', 'mutation_type','reference_genome_allele',
    'mutated_from_allele','mutated_to_allele','specimen_type', 'tumour_confirmed']]= df[['icgc_donor_id', 'donor_sex','donor_vital_status','disease_status_last_followup',
    'donor_diagnosis_icd10', 'prior_malignancy','cancer_history_first_degree_relative','mutation_type1',
    'assembly_version','sequencing_strategy', 'mutation_type','reference_genome_allele',
    'mutated_from_allele','mutated_to_allele','specimen_type', 'tumour_confirmed']].apply(LabelEncoder().fit_transform)
df.to_csv('C:/python39/BRCA/preprocessed data/encoded.csv', index=False)


#applying MICE Imputation technique to fill the missing values on every file

df = pd.read_csv('C:/python39/BRCA/preprocessed data/encoded.csv' , encoding_errors= 'replace')
# calling the MICE class
mice_imputer = IterativeImputer()
# imputing the missing value with mice imputer
imputer = mice_imputer.fit_transform(df)
df_imputed = pd.DataFrame(imputer)
df_imputed.to_csv('C:/python39/BRCA/preprocessed data/mice_imputed.csv', index=False)"""

#selecting features using Randomforest elimination method


df = pd.read_csv('mice_imputed.csv', encoding_errors= 'replace')

# import Random Forest classifier

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.3,random_state=0)

print(X_train.shape, X_test.shape)

# feature extraction
#model = LogisticRegression(solver='lbfgs')
rfe = RFE(estimator=RandomForestRegressor(), n_features_to_select=5)
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
fit = rfe.fit(X, y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

# evaluate model
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=42)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=5, n_jobs=-1, error_score='raise')
# Print the results
print("Optimal number of features: %d" % model.n_features_)
print("Feature ranking:", model.ranking_)

# Print accuracy scores for each feature subset
for i, score in enumerate(model.cv_results_['mean_test_score']):
    print("Number of features:", i + 1, "| Accuracy Score:", score)

# Get the selected features
selected_features = [f for f, s in zip(range(len(model.support_)), model.support_) if s]

# report performance
print('Residual: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
#.to_csv('C:/python39/BRCA/preprocessed data/selected_features.csv', index=False)




