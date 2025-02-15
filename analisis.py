import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
import statsmodels.stats.diagnostic as diag
import statsmodels.api as sm
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm

def trans_categorical(df):
    # castWomen y castMen parsing
    df['castWomenAmount'] = pd.to_numeric(df['castWomenAmount'], errors='coerce')
    df['castMenAmount'] = pd.to_numeric(df['castMenAmount'], errors='coerce')
    df['castWomenAmount'] = df['castWomenAmount'].astype(float)
    df['castMenAmount'] = df['castMenAmount'].astype(float)
    df = df.dropna(subset=['castWomenAmount', 'castMenAmount'])
 
    # Promedio de popularidad:
    df = df[df['revenue']!=0]
    df['actorsPopularityAvg'] = df['actorsPopularity'].astype(str).apply(lambda x: sum(map(float, x.split('|'))) / len(x.split('|')))
    df['actorsPopularityMax'] = df['actorsPopularity'].astype(str).apply(lambda x: max(map(float, x.split('|'))))
    df['actorsPopularityMin'] = df['actorsPopularity'].astype(str).apply(lambda x: min(map(float, x.split('|'))))
    # Extracción de Mes
    df['releaseMonth'] = pd.to_datetime(df['releaseDate']).dt.month
    df['releaseYear'] = pd.to_datetime(df['releaseDate']).dt.year
    df['releaseWeekDay'] = pd.to_datetime(df['releaseDate']).dt.weekday
    
    # Video y HomePage a 0 y 1
    df['video'] = df['video'].apply(lambda x: 0 if pd.isna(x) or x is False else 1)
    df['homePage'] = df['homePage'].apply(lambda x: 0 if pd.isna(x) or x is False else 1)
    return df

def preprocesses(df):
    # Eliminación de datos 0 (se sabe que todos serán mayor a 0)
    nmovies = df.dropna(axis=0)
    cols_to_check = nmovies.columns.difference(["video", "homePage"])
    nfmovies = nmovies[(nmovies[cols_to_check] > 0).all(axis=1)]
    nfmovies.describe()
    
    # Eliminación de atípicos con IQR
    Q1 = nfmovies.quantile(0.25)  # 25th percentile
    Q3 = nfmovies.quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile range


    v = nfmovies[~((nfmovies < (Q1 - 1.5 * IQR)) | (nfmovies > (Q3 + 1.5 * IQR))).any(axis=1)]
    v = v[['popularity','voteAvg','budget','revenue']]
    return v

def elbow(X_scale):
    random.seed(123)
    numeroClusters = range(1,20)
    wcss = []
    for i in numeroClusters:
        kmeans = sklearn.cluster.KMeans(n_clusters=i)
        kmeans.fit(X_scale)
        wcss.append(kmeans.inertia_)

    plt.plot(numeroClusters, wcss, marker='o')
    plt.xticks(numeroClusters)
    plt.xlabel("K clusters")
    plt.ylabel("WSS")
    plt.title("Gráfico de Codo")
    plt.show()
    
def plotFeatures(X, cluster_n):
    for i in range(0,cluster_n):
        cl0 = X[X['Cluster']==i]
        fi, ax = plt.subplots(ncols=2,nrows=2, figsize=(12,8))
        fi.suptitle(f'Características de Cluster {i+1}', fontsize=14, fontweight='bold')  # Overarching title
        ax[0,0].set_title(f'Histograma Budget')
        ax[0,0].hist(cl0['budget'],bins=20)
        ax[1,0].set_title(f'Histograma Revenue')
        ax[1,0].hist(cl0['revenue'],bins=20)
        ax[0,1].set_title(f'Histograma Popularity')
        ax[0,1].hist(cl0['popularity'],bins=20)
        ax[1,1].set_title(f'Histograma Vote Average{i}')
        ax[1,1].hist(cl0['voteAvg'],bins=20)
        plt.show()
        
def sillhouette(range_n_clusters, X):

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()