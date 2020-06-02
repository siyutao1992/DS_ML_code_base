###############################################################################################
### Machine learning -- unsupervised (plus supervised nnet model)
import numpy as np
import pandas as pd

###############################################################################################
#### Unsupervised learning models

#####################################
## k-means model
# standardize data for better result
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
samples_scaled = scaler.transform(samples)
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples_scaled) # samples are a 2D matrix
labels = model.predict(samples_scaled) # labels for existing observations
new_labels = model.predict(samples_scaled)
print(model.inertia_) # minimize inertia as a metric for model quality
#   plot the inertia as a function of n_clusters parameter
#   choose the "elbow" point as the best model

#####################################
## Hierarchical clustering (result is a dendrogram)
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
mergings = linkage(samples_scaled, method='complete')
dendrogram(mergings, labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()
#   Height on dendrogram = distance between merging clusters
# can further assign cluster labels based on above result
from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 15, criterion='distance')
pairs = pd.DataFrame({'labels': labels, 'countries': country_names})
print(pairs.sort_values('labels'))

#####################################
## t-SNE for 2D map visualizing high-dim data
from sklearn.manifold import TSNE
model = TSNE(learning_rate=100) # learning rate is usually within [50, 200]
transformed = model.fit_transform(samples)
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs, ys, c=species)
plt.show()

#####################################
## PCA transformation
from sklearn.decomposition import PCA
model = PCA(n_components=2) # set n_component to None to retain all features
model.fit(samples)
transformed = model.transform(samples)
#   Rows of transformed correspond to samples
#   Columns of transformed are the "PCA features"
print(model.components_)
# visualize explained variance by each component
features = range(model.n_components_)
plt.bar(features, model.explained_variance_)


###############################################################################################
#### nnet 

#####################################
## keras - regression
from keras.layers import Dense
from keras.models import Sequential
predictors = np.loadtxt('predictors_data.csv', delimiter=',')
target = np.loadtxt('target_data.csv', delimiter=',')
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(predictors, target, validation_split=0.3, epochs=20, callbacks = [early_stopping_monitor])

## keras - classification
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
data = pd.read_csv('basketball_shot_log.csv')
predictors = data.drop(['shot_result'], axis=1).as_matrix()
target = to_categorical(data.shot_result)
n_cols = predictors.shape[1]
model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer=SGD(lr=lr), loss='categorical_crossentropy') # SGD optimization
model.fit(predictors, target, validation_split=0.3, epochs=20, callbacks = [early_stopping_monitor])

## keras - save and load model
# for more details, please refer to datacamp course "Deep Learning in Python"
my_model = load_model('my_model.h5')
my_model.summary() # verify the model structure


