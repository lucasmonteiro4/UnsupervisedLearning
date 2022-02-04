# this work is a personal work
# I explore by myself some techniques of clustering (kmeans, dbscan, affiniity propagation, spectral clustering)
# I work on a dataset "crime" which registered several kinds of crime and just parctice with it

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA, KernelPCA
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AffinityPropagation
from kneed import KneeLocator

crime = pd.read_csv('crime.csv')

crime.head()
features = crime.drop('State', 1)
features.corr()
sns.heatmap(features.corr(), annot=True, vmin=-1)
plt.show()

# naturally, features are strongly correlated

features = features.drop('Population', 1)
scaler = StandardScaler()
dfnorm = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)
dfnorm.head()

# we will considered 2 ways to dimensional reduction : PCA and KernelPCA(nonlinear)

#PCA

pca = PCA(n_components=2)
pca.fit(dfnorm)
dfpca_norm = pd.DataFrame(pca.fit_transform(dfnorm))
dfpca_norm.head()
norm_eigenvalues = pca.explained_variance_
print(norm_eigenvalues)
pca_columns = ['PC' + str(i+1) for i in range(pca.n_components_)]
dfpca_norm.columns = pca_columns
dfpca_norm.head()
sns.scatterplot(x = "PC1", y= "PC2", data = dfpca_norm)
plt.show()
reduced_data = pca.fit_transform(dfnorm)


#KernelPCA
kpca = KernelPCA(n_components=2) #reconstruction coef high enough 
kpca.fit(dfnorm)
dfkpca_norm = pd.DataFrame(kpca.fit_transform(dfnorm))
dfkpca_norm.head()

kpca.lambdas_
kpca.eigenvectors_

kpca_columns = ['KernelPC' + str(i+1) for i in range(len(kpca.eigenvalues_))]
dfkpca_norm.columns = kpca_columns
dfkpca_norm.head()
sns.scatterplot(x = "KernelPC1", y= "KernelPC2", data = dfkpca_norm)
plt.show()
kreduced_data = kpca.fit_transform(dfnorm)

# we make clustering techniques for reduced_pca ( sparse_pca and kernel_pca at the end)
kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(reduced_data)
    sse.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
print(sse)

# we see on the plot that it is not easy to choose the right number

kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
kl.elbow
# we go with 3 clusters
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(reduced_data)

# The lowest SSE value
kmeans.inertia_

kmeans.cluster_centers_
kmeans.n_iter_
kmeans.labels_


dbscan = DBSCAN(eps=0.3)
dbscan.fit(reduced_data)

af = AffinityPropagation(preference=-50, random_state=0).fit(reduced_data)
af.cluster_centers_
af.n_iter_
af.labels_

from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

kmeans_silhouette = silhouette_score(reduced_data, kmeans.labels_).round(2)
dbscan_silhouette = silhouette_score(reduced_data, dbscan.labels_).round(2)
af_silhouette = silhouette_score(reduced_data, af.labels_).round(2)
kmeans_silhouette
dbscan_silhouette
af_silhouette

kmeans_silhouette_m = silhouette_score(reduced_data, kmeans.labels_, metric  ="mahalanobis").round(2)
dbscan_silhouette_m = silhouette_score(reduced_data, dbscan.labels_, metric  ="mahalanobis").round(2)
af_silhouette_m = silhouette_score(reduced_data, af.labels_, metric  ="mahalanobis").round(2)
kmeans_silhouette_m
dbscan_silhouette_m
af_silhouette_m

import numpy as np
metrics.calinski_harabasz_score(reduced_data, kmeans.labels_)
metrics.calinski_harabasz_score(reduced_data, dbscan.labels_)
metrics.calinski_harabasz_score(reduced_data, af.labels_)

# if the ground truth labels are not known, the Calinski-Harabasz index- also known as the Variance Ratio Criterion - 
# can be used to evaluate the model, where a higher Calinski-Harabasz score relates to a model with better defined clusters.

metrics.davies_bouldin_score(reduced_data, kmeans.labels_)
metrics.davies_bouldin_score(reduced_data, dbscan.labels_)
metrics.davies_bouldin_score(reduced_data,  af.labels_)

# If the ground truth labels are not known, the Davies-Bouldin index can be used to evaluate the model, 
# where a lower Davies-Bouldin index relates to a model with better separation between the clusters.

# so in our case, we see that kmeans seems to produce better definition and separation between clusters


reduced_data = pd.DataFrame(reduced_data)
reduced_data.columns = ["PC1", "PC2"]


sns.scatterplot(x = "PC1", y = "PC2", data = reduced_data, hue = kmeans.labels_ , palette = 'Dark2')
plt.show()

sns.scatterplot(x = "PC1", y = "PC2", data = reduced_data, hue = dbscan.labels_ , palette = 'rainbow')
plt.show()

sns.scatterplot(x = "PC1", y = "PC2", data = reduced_data, hue = af.labels_ , palette = 'rainbow')
plt.show()

spec = SpectralClustering(n_clusters=3,random_state=42)


spec.fit(reduced_data)
spec.labels_

sns.scatterplot(x = "PC1", y = "PC2", data = reduced_data, hue = spec.labels_ , palette = 'rainbow')
plt.show()







# Kernel PCA
sse_k = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(kreduced_data)
    sse_k.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(1, 11), sse_k)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE kernel")
plt.show()
print(sse_k)

# we see on the plot that it is not easy to choose the right number

kl = KneeLocator(range(1, 11), sse_k, curve="convex", direction="decreasing")
kl.elbow
# we go with 3 clusters
kmeans_k = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans_k.fit(kreduced_data)

# The lowest SSE value
kmeans_k.inertia_

kmeans_k.cluster_centers_
kmeans_k.n_iter_
kmeans_k.labels_


dbscan_k = DBSCAN(eps=0.3)
dbscan_k.fit(kreduced_data)

af_k = AffinityPropagation(preference=-50, random_state=0).fit(kreduced_data)
af_k.cluster_centers_
af_k.n_iter_
af_k.labels_

from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

kmeans_k_silhouette = silhouette_score(kreduced_data, kmeans_k.labels_).round(2)
dbscan_k_silhouette = silhouette_score(kreduced_data, dbscan_k.labels_).round(2)
af_k_silhouette = silhouette_score(kreduced_data, af_k.labels_).round(2)
kmeans_k_silhouette
dbscan_k_silhouette
af_k_silhouette


metrics.calinski_harabasz_score(kreduced_data, kmeans_k.labels_)
metrics.calinski_harabasz_score(kreduced_data, dbscan_k.labels_)
metrics.calinski_harabasz_score(kreduced_data, af_k.labels_)


metrics.davies_bouldin_score(kreduced_data, kmeans_k.labels_)
metrics.davies_bouldin_score(kreduced_data, dbscan_k.labels_)
metrics.davies_bouldin_score(kreduced_data,  af_k.labels_)

# If the ground truth labels are not known, the Davies-Bouldin index can be used to evaluate the model, 
# where a lower Davies-Bouldin index relates to a model with better separation between the clusters.

# so in our case, we see that kmeans seems to produce better definition and separation between clusters


kreduced_data = pd.DataFrame(kreduced_data)
kreduced_data.columns = ["KernPC1", "KernPC2"]


sns.scatterplot(x = "KernPC1", y = "KernPC2", data = kreduced_data, hue = kmeans_k.labels_ , palette = 'Dark2')
plt.show()

sns.scatterplot(x = "KernPC1", y = "KernPC2", data = kreduced_data, hue = dbscan_k.labels_ , palette = 'rainbow')
plt.show()

sns.scatterplot(x = "KernPC1", y = "KernPC2", data = kreduced_data, hue = af_k.labels_ , palette = 'rainbow')
plt.show()

# globally the result are the same, however we see a small diffenrence concerning dbscan method with PCA and kernelPCA
# the dataset is not really good for this kind of analysis, but for an introduction, it is more or less relevant
