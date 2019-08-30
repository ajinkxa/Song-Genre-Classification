#!/usr/bin/env python
# coding: utf-8

# This project is taken from datacamp website, where the basics are explained in great detail. What I have added here is further explanation of the code, and then some advanced models to make predictions more accurate.

# ## 1. Preparing our dataset
# <p><em>These recommendations are so on point! How does this playlist know me so well?</em></p>
# <p><img src="https://s3.amazonaws.com/assets.datacamp.com/production/project_449/img/iphone_music.jpg" alt="Project Image Record" width="600px"></p>
# <p>Over the past few years, streaming services with huge catalogs have become the primary means through which most people listen to their favorite music. But at the same time, the sheer amount of music on offer can mean users might be a bit overwhelmed when trying to look for newer music that suits their tastes.</p>
# <p>For this reason, streaming services have looked into means of categorizing music to allow for personalized recommendations. One method involves direct analysis of the raw audio information in a given song, scoring the raw data on a variety of metrics. Today, we'll be examining data compiled by a research group known as The Echo Nest. Our goal is to look through this dataset and classify songs as being either 'Hip-Hop' or 'Rock' - all without listening to a single one ourselves. In doing so, we will learn how to clean our data, do some exploratory data visualization, and use feature reduction towards the goal of feeding our data through some simple machine learning algorithms, such as decision trees and logistic regression.</p>
# <p>To begin with, let's load the metadata about our tracks alongside the track metrics compiled by The Echo Nest. A song is about more than its title, artist, and number of listens. We have another dataset that has musical features of each track such as <code>danceability</code> and <code>acousticness</code> on a scale from -1 to 1. These exist in two different files, which are in different formats - CSV and JSON. While CSV is a popular file format for denoting tabular data, JSON is another common file format in which databases often return the results of a given query.</p>
# <p>Let's start by creating two pandas <code>DataFrames</code> out of these files that we can merge so we have features and labels (often also referred to as <code>X</code> and <code>y</code>) for the classification later on.</p>

# In[2]:


import pandas as pd

# Read in track metadata with genre labels
tracks = pd.read_csv("fma-rock-vs-hiphop.csv")

# Read in track metrics with the features
echonest_metrics = pd.read_json("echonest-metrics.json", precise_float = True)

# Merge the relevant columns of tracks and echonest_metrics
echo_tracks = pd.merge(echonest_metrics, tracks[["track_id", "genre_top"]], on="track_id")

#Here, we are merging two datasets- echonest_metrics and tracks, but we are retaining only two columns from tracks dataset - track_id and genre_top.
#We are merging the dataset based on track_id

# Inspect the resultant dataframe
echo_tracks.info()
echo_tracks.describe()


# precise_float is a parameter set to enable usage of higher precision (strtod) function when decoding string to double values. Default (False) is to use fast but less precise builtin functionality. For better understanding, here is a great article demonstrating the strtod function. (https://www.geeksforgeeks.org/strtod-function-in-c-c/)

# ## 2. Pairwise relationships between continuous variables
# <p>We typically want to avoid using variables that have strong correlations with each other -- hence avoiding feature redundancy -- for a few reasons:</p>
# <ul>
# <li>To keep the model simple and improve interpretability (with many features, we run the risk of overfitting).</li>
# <li>When our datasets are very large, using fewer features can drastically speed up our computation time.</li>
# </ul>
# <p>To get a sense of whether there are any strongly correlated features in our data, we will use built-in functions in the <code>pandas</code> package.</p>

# In[3]:


# Create a correlation matrix
corr_metrics = echo_tracks.corr()
corr_metrics.style.background_gradient()


# Put in plain terms, correlation is a measure of how strongly one variable depends on another.
# 
# Correlation can be an important tool for feature engineering in building machine learning models. Predictors which are uncorrelated with the objective variable are probably good candidates to trim from the model (shoe size is not a useful predictor for salary). In addition, if two predictors are strongly correlated to each other, then we only need to use one of them (in predicting salary, there is no need to use both age in years, and age in months). Taking these steps means that the resulting model will be simpler, and simpler models are easier to interpret.
# 
# Refer to this great article (https://blog.bigml.com/2015/09/21/looking-for-connections-in-your-data-correlation-coefficients/) for further explanation on correlation.

# ## 3. Normalizing the feature data
# <p>As mentioned earlier, it can be particularly useful to simplify our models and use as few features as necessary to achieve the best result. Since we didn't find any particular strong correlations between our features, we can instead use a common approach to reduce the number of features called <strong>principal component analysis (PCA)</strong>. </p>
# 
# Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.
# 
# Here is a very nice video explaining PCA in detail (https://www.youtube.com/watch?v=FgakZw6K1QQ&vl=en).
# 
# <p>It is possible that the variance between genres can be explained by just a few features in the dataset. PCA rotates the data along the axis of highest variance, thus allowing us to determine the relative contribution of each feature of our data towards the variance between classes. </p>
# <p>However, since PCA uses the absolute variance of a feature to rotate the data, a feature with a broader range of values will overpower and bias the algorithm relative to the other features. To avoid this, we must first normalize our data. There are a few methods to do this, but a common way is through <em>standardization</em>, such that all features have a mean = 0 and standard deviation = 1 (the resultant is a z-score).</p>

# In[4]:


# Define our features 
features = echo_tracks.drop(["genre_top","track_id"], axis=1)

# Define our labels
labels = echo_tracks["genre_top"]

# Import the StandardScaler
from sklearn.preprocessing import StandardScaler

# Scale the features and set the values to a new variable
scaler = StandardScaler()
scaled_train_features = scaler.fit_transform(features)


# ## 4. Principal Component Analysis on our scaled data
# <p>Now that we have preprocessed our data, we are ready to use PCA to determine by how much we can reduce the dimensionality of our data. We can use <strong>scree-plots</strong> and <strong>cumulative explained ratio plots</strong> to find the number of components to use in further analyses.</p>
# <p>Scree-plots display the number of components against the variance explained by each component, sorted in descending order of variance. Scree-plots help us get a better sense of which components explain a sufficient amount of variance in our data. When using scree plots, an 'elbow' (a steep drop from one data point to the next) in the plot is typically used to decide on an appropriate cutoff.</p>

# In[5]:


# This is just to make plots appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Import our plotting module, and PCA class
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Get our explained variance ratios from PCA using all features
pca = PCA()
pca.fit(scaled_train_features)
exp_variance = pca.explained_variance_ratio_

print(pca.explained_variance_ratio_)
print(pca.n_components_)

# plot the explained variance using a barplot
fig, ax = plt.subplots()
ax.bar(range(8), exp_variance)
ax.set_xlabel('Principal Component #')


# ## 5. Further visualization of PCA
# <p>Unfortunately, there does not appear to be a clear elbow in this scree plot, which means it is not straightforward to find the number of intrinsic dimensions using this method. </p>
# <p>But all is not lost! Instead, we can also look at the <strong>cumulative explained variance plot</strong> to determine how many features are required to explain, say, about 90% of the variance (cutoffs are somewhat arbitrary here, and usually decided upon by 'rules of thumb'). Once we determine the appropriate number of components, we can perform PCA with that many components, ideally reducing the dimensionality of our data.</p>

# In[6]:


# Import numpy
import numpy as np

# Calculate the cumulative explained variance
cum_exp_variance = np.cumsum(exp_variance)

# Plot the cumulative explained variance and draw a dashed line at 0.90.
fig, ax = plt.subplots()
ax.plot(range(8), cum_exp_variance)
ax.axhline(y=0.9, linestyle='--')
n_components = 6

# Perform PCA with the chosen number of components and project data onto components
pca = PCA(n_components, random_state=10)
pca.fit(scaled_train_features)
pca_projection = pca.transform(scaled_train_features)


# ## 6. Balance our data for greater performance
# <p>Both our models do similarly well, boasting an average precision of 87% each. However, looking at our classification report, we can see that rock songs are fairly well classified, but hip-hop songs are disproportionately misclassified as rock songs. </p>
# <p>Why might this be the case? Well, just by looking at the number of data points we have for each class, we see that we have far more data points for the rock classification than for hip-hop, potentially skewing our model's ability to distinguish between classes. This also tells us that most of our model's accuracy is driven by its ability to classify just rock songs, which is less than ideal.</p>
# <p>To account for this, we can weight the value of a correct classification in each class inversely to the occurrence of data points for each class. Since a correct classification for "Rock" is not more important than a correct classification for "Hip-Hop" (and vice versa), we only need to account for differences in <em>sample size</em> of our data points when weighting our classes here, and not relative importance of each class. </p>

# In[7]:


# Subset only the hip-hop tracks, and then only the rock tracks
hop_only = echo_tracks.loc[echo_tracks["genre_top"] == "Hip-Hop"]
rock_only = echo_tracks.loc[echo_tracks["genre_top"] == "Rock"]

# sample the rocks songs to be the same number as there are hip-hop songs
rock_only = rock_only.sample(len(hop_only), random_state=10)

# concatenate the dataframes rock_only and hop_only
rock_hop_bal = pd.concat([rock_only, hop_only])

# The features, labels, and pca projection are created for the balanced dataframe
features = rock_hop_bal.drop(['genre_top', 'track_id'], axis=1) 
labels = rock_hop_bal['genre_top']
pca_projection = pca.fit_transform(scaler.fit_transform(features))

# Import train_test_split function and Decision tree classifier
from sklearn.model_selection import train_test_split

# Define the train and test set with the pca_projection from the balanced data
train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)


# ## 7. Understanding the Evaluation Metrics
# 
# Before we move onto the algorithms, it is crucial to understand the metric we are going to use to evaluate our predictive models.
# 
# Accuracy - Accuracy is the most intuitive performance measure and it is simply a ratio of correctly predicted observation to the total observations. One may think that, if we have high accuracy then our model is best. Yes, accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same. Therefore, you have to look at other parameters to evaluate the performance of your model. For our model, we have got 0.803 which means our model is approx. 80% accurate.
# 
# Accuracy = TP+TN/TP+FP+FN+TN
# 
# Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate. We have got 0.788 precision which is pretty good.
# 
# Precision = TP/TP+FP
# 
# Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. The question recall answers is: Of all the passengers that truly survived, how many did we label? We have got recall of 0.631 which is good for this model as it’s above 0.5.
# 
# Recall = TP/TP+FN
# 
# F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall. In our case, F1 score is 0.701.
# 
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)
# 
# As you may have got an idea, the evaluation metric that we are going to use is Accuracy. The evaluation metric depends on the type of data that you are analyzing. If we were handling a sensitive problem (like cancer test, loan application) then false positives matter a lot. But, we are only dealing with songs, and no one label is more crucia or sensitive than the other. Hence, accuracy is what we will be more concerned with.

# ## 8. Applying the Algorithms
# 
# This is the part where we start creating different predictive models and evaluate them for their accuracy. I'll also breifly mention the mathematics behind the model and provide reference for further information.

# ### 8.1 KNN

# In[9]:


# Create the classification report for all the models we shall use
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


#Importing a package to fine-tune and optimize our algorithms
from sklearn.model_selection import GridSearchCV
knn_grid_params = {
    'n_neighbors': [1,2,3,4,5,6,7],
    'weights': ['uniform','distance'],
    'metric': ['euclidean','manhattan']
}
knn_gs = GridSearchCV(KNeighborsClassifier(),knn_grid_params,verbose=1, n_jobs = -1)
knn_gs_results = knn_gs.fit(train_features, train_labels)
print("Accuracy of the KNN model is: ", knn_gs_results.best_score_)
print("KNN model best parameters are: ", knn_gs_results.best_params_)


# K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases based on a similarity measure (e.g., distance functions).
# 
# Reference: https://medium.com/@erikgreenj/k-neighbors-classifier-with-gridsearchcv-basics-3c445ddeb657

# ### 8.2 Logistic Regression 

# In[20]:


# Train our logistic regression on the balanced data
from sklearn.linear_model import LogisticRegression
logit_grid_params = {
    'C':[0.001,0.1,1,10,100],
    'penalty':['l1','l2','elasticnet'],
    'solver':['saga'],
    'l1_ratio':[0,0.1,0.5,0.7,1]
}
logit_gs = GridSearchCV(LogisticRegression(),logit_grid_params,verbose=1, n_jobs = -1)
logit_gs_results = logit_gs.fit(train_features, train_labels)
print("Accuracy of the Logistic Regression model is: ", logit_gs_results.best_score_)
print("Logistic Regression model best parameters are: ", logit_gs_results.best_params_)


# I am not quite sure about the parameters used in this algorithm. I am researching on it to find exactly what does each parameter specify and how it affects the predictive capabilities of the model.

# ### 8.3 Decision Trees

# In[34]:


# Train our decision tree on the balanced data
from sklearn.tree import DecisionTreeClassifier
tree_grid_params = {
    'criterion': ['gini','entropy'],
    'min_samples_split': [2,3,4,5,6,7,8,9,10],
    'min_samples_leaf' : [1,2,3,4],
}
tree_gs = GridSearchCV(DecisionTreeClassifier(),tree_grid_params,verbose=1, n_jobs = -1)
tree_gs_results = tree_gs.fit(train_features, train_labels)
print("Accuracy of the Decision Tree Classifier model is: ", tree_gs_results.best_score_)
print("Decision Tree Classifier model best parameters are: ", tree_gs_results.best_params_)


# As we can see, decision tree classifier gives us a very low accuracy. The best from the models we evaluated so far is the KNN.

# ### 8.4 Random Forest

# In[42]:


# Train our random forest on the balanced data
from sklearn.ensemble import RandomForestClassifier
rf_grid_params = {
    'n_estimators': [60,70,80,90,100],
    'criterion': ['gini','entropy']
}
rf_gs = GridSearchCV(RandomForestClassifier(),rf_grid_params,verbose=1, n_jobs = -1)
rf_gs_results = rf_gs.fit(train_features, train_labels)
print("Accuracy of the Random Forest Classifier model is: ", rf_gs_results.best_score_)
print("Random Forest Classifier model best parameters are: ", rf_gs_results.best_params_)


# ### 8.5 xG boost

# In[49]:


# Train our XG Boost model on the balanced data
from sklearn.ensemble import GradientBoostingClassifier
gb_grid_params = {
    'loss': ['deviance','exponential'],
    'learning_rate': [0.01,0.04,0.1,0.2,0.5],
    'n_estimators': [60,70,80,90,100],
    'warm_start': ['True','False']
}
gb_gs = GridSearchCV(GradientBoostingClassifier(),gb_grid_params,verbose=1, n_jobs = -1)
gb_gs_results = gb_gs.fit(train_features, train_labels)
print("Accuracy of the Gradient Boosting Classifier model is: ", gb_gs_results.best_score_)
print("Gradient Boosting Classifier model best parameters are: ", gb_gs_results.best_params_)


# ### 8.6 Support Vector Machine

# In[52]:


# Train our SVM model on the balanced data
from sklearn.svm import SVC
svc_grid_params = {
    'C':[0.001,0.1,1,10,100],
    'degree':[1,2,3,4,5],
    }
svc_gs = GridSearchCV(SVC(),svc_grid_params,verbose=1, n_jobs = -1)
svc_gs_results = svc_gs.fit(train_features, train_labels)
print("Accuracy of the SVC model is: ", svc_gs_results.best_score_)
print("SVC model best parameters are: ", svc_gs_results.best_params_)


# ### 8.7 Neural Networks

# In[56]:


# Train our neural network model on balanced data
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                     solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
mlp_results = mlp.fit(train_features, train_labels)
pred = mlp.predict(train_features)
accuracy_score(train_labels, pred)


# I am not sure this is the best implementation of neural network model. I just tried to build a very basic model and I can already see it's not too efficient. But, the good thing is we now have a more accurate model, which means that neural network are a great tool to solve our problem. I plan to make a separate kernel to build a neural network model and fine-tune it to make it more efficient.
