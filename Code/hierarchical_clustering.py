#%% Imports

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

import scipy.cluster.hierarchy as hierarchy

from sklearn.decomposition import PCA


#%% Define useful functions

def compute_clusters(Z, k=5) :
    clusters = hierarchy.fcluster(Z, k, criterion='maxclust')
    clusters -= 1
    return clusters

def compute_pca(X, n_components=2) :
    pca = PCA(n_components)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents,
                               columns = ['x1', 'x2'])
    return principalDf

def plot_clusters(clusters, principalDf) :
    plt.figure(figsize=(10, 8))
    plt.scatter(principalDf.x1, principalDf.x2,
                c=clusters, cmap='prism') #plot points with cluster dependent colors
    plt.show()
    
def plot_dendrogram(Z) :
    dn = hierarchy.dendrogram(Z)
    hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
                               orientation='top')
    dn2 = hierarchy.dendrogram(Z, ax=axes[1],
                               above_threshold_color='#bcbddc',
                               orientation='right')
    hierarchy.set_link_color_palette(None)  # reset to default after use
    plt.show()
    
#%% Compute hierarchical linkage

df=pd.read_csv('C:\\Users\\charl\\Documents\\3A\\MAP573\\data_preprocessed_missForest.csv')
df.rename(columns={'Death': 'Yfact', 'Tranexamic.acid': 'Tr'}, inplace=True)

# Create dataframe for clustering
X = df.copy()
del X['Tr']
del X['Yfact']

Z = hierarchy.linkage(X, method='ward')

#%% Compute and plot clusters

min_clusters = 2
max_clusters = 25

clusters = np.array([compute_clusters(Z,k) for k in range(min_clusters,max_clusters+1)]).T

# Plot clusters
k = 3 #number of clusters to plot
n_components = 2
principalDf = compute_pca(X,n_components)
plot_clusters(clusters[:,k-min_clusters], principalDf)


#%% Export clusters as csv

clusters = pd.DataFrame(clusters)
clusters.columns = [str(i)+'_clusters' for i in range(min_clusters,max_clusters+1)]

clusters.to_csv('hierarchical_clusters.csv')