#!/usr/bin/env python
# coding: utf-8

# # </html>
# <h1 style="text-align:center;color:red;background-color:powderred;font-size:500%">Resume Shortlister</h1>
# <html>

# .

# .

# .

# # <html>
# <img src="https://s.wsj.net/public/resources/images/OG-DN954_201912_GR_20191213124716.gif" style="width:1000px;height:600px;"/>
# </html>

# ,

# # Importing Liabraries

# In[12]:


import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
import PyPDF2
import warnings 
warnings.filterwarnings('ignore')


# In[13]:


desired_skills = ['machine learning', 'data analysis', 'python', 'statistics']


# In[14]:


import PyPDF2

resumes_df = pd.DataFrame(columns=['filename', 'text'])
for filename in ['CV___Adarsh_Tayade (3).pdf','Anuradha Resume.pdf','Aliasgar Resume.pdf','Aditya  Resume (1).pdf','Darshan Resume data Science.pdf',
                'Gauri D. Auti Resume .pdf','College_Resume.pdf','Ishwari Resume.pdf','Magadh Resume updated.pdf','Poorvi Resume.pdf']:
    with open(filename, 'rb') as doc:
        pdf_reader = PyPDF2.PdfReader(doc)
        text = ""
        for i in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[i]
        
            # Extract the text from the page
            page_text = page.extract_text()
        
            # Add the page text to the overall text variable
            text += page_text
        
        resumes_df = resumes_df.append({'filename': filename, 'text': text}, ignore_index=True)


# In[15]:


resumes_df['word_count'] = resumes_df['text'].apply(lambda x: len(x.split()))
resumes_df['char_count'] = resumes_df['text'].apply(lambda x: len(x))
resumes_df['education_level'] = resumes_df['text'].apply(lambda x: ('Bachelor' if re.search('Bachelor', x, re.IGNORECASE) else 'Unknown'))
resumes_df['work_experience'] = resumes_df['text'].apply(lambda x: re.search(r'\d+ years of experience', x, re.IGNORECASE).group(0)[:-16] if re.search(r'\d+ years of experience', x, re.IGNORECASE) else 0)


# In[16]:


vectorizer = TfidfVectorizer(vocabulary=desired_skills, lowercase=True)
skills_counts = vectorizer.fit_transform(resumes_df['text'])
skills_df = pd.DataFrame(skills_counts.toarray(), columns=vectorizer.vocabulary_)
resumes_df = pd.concat([resumes_df, skills_df], axis=1)


# In[17]:


feature_cols = ['word_count', 'char_count', 'education_level', 'work_experience'] + list(desired_skills) 


# In[18]:


def calculate_resume_score(row):
    score = 0
    for skill in desired_skills:
        if row[skill] == 1:
            score += 1
    score += row['word_count'] / 100
    score += row['char_count'] / 1000
    if row['education_level'] == 'Bachelor':
        score += 1
    score += int(row['work_experience'])
    return score


# In[19]:


resumes_df['score'] = resumes_df.apply(calculate_resume_score, axis = 1)


# In[20]:


# Oversample minority class
scores = resumes_df['score']
X_resampled, scores_resampled = resample(skills_counts, scores, replace = True, n_samples = 10, random_state = 42)


# In[21]:


# define the number of bins
num_bins = 5
# use pd.qcut to create categorical bins for the scores based on quantiles
resumes_df['score'] = pd.qcut(resumes_df['score'], q=num_bins, labels=False, duplicates='drop')
# get the categories as integers
scores_resampled = np.array(resumes_df['score'])


# In[22]:


svm_param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'sigmoid']}
dt_param_grid = {'max_depth': [2, 5, 10], 'min_samples_split': [2, 5, 10]}
svm_grid = GridSearchCV(SVC(), svm_param_grid, cv=2, n_jobs=-1)
dt_grid = GridSearchCV(DecisionTreeClassifier(), dt_param_grid, cv=2, n_jobs=-1)
svm_grid.fit(X_resampled, scores_resampled)
dt_grid.fit(X_resampled, scores_resampled)
svm_best = svm_grid.best_estimator_
dt_best = dt_grid.best_estimator_


# In[23]:


rf_best = RandomForestClassifier(max_depth = 10, min_samples_split = 10, n_estimators = 100, random_state = 42)
estimators = [('svm', svm_best), ('dt', dt_best), ('rf', rf_best)]
voting_classifier = VotingClassifier(estimators)


# In[24]:


voting_classifier.fit(X_resampled, scores_resampled)


# In[25]:


resumes_df['predicted_score'] = voting_classifier.predict(skills_counts)


# In[27]:


top_resumes = resumes_df.sort_values('predicted_score', ascending = False).head(2)
print(top_resumes[['filename', 'score', 'predicted_score']])


# In[ ]:




