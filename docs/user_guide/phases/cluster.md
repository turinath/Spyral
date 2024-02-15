# The Second Phase: Clustering

Once point clouds have been created in the first phase, the next phase aims to break the point clouds into clusters. Clusters are collections of points which are presumed to belong to a single particle trajectory. Clustering is a very common form of unsupervised machine learning, clustering takes a dataset and applies labels to samples (points) based on the features (coordinates, charge). In this phase we make use of the [scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#clustering) algorithms.

There are three steps to this phase:

1. Initial clustering using HDBSCAN
2. Joining of clusters using circle-fitting and relative charge
3. Cleanup of joined clusters

Below we'll break down each of these steps

## Initial Clustering with HDBSCAN

In some ways AT-TPC data doesn't fit very neatly into many of the clustering algorithms. The data is in general noisy, trajectories are not isolated spatially, and the density of the points can vary within a single trajectory due to the small/large pads on the pad plane. Additionally, there can be gaps in the trajectory due to the hole in the pad plane. Thankfully, there is an algorithm, [HDBSCAN](https://scikit-learn.org/stable/modules/clustering.html#hdbscan) which covers most of our bases.

HDBSCAN is a modified DBSCAN algorithm; DBSCAN stands for Density-Based Spatial Clustering of Applications wth Noise, and is a powerful clustering algorithm for noisy data. It clusters points based on their density (closeness to neighbors), but it assumes a *global* density. This means that DBSCAN assumes all clusters have the same density of points. As we mentioned before, this is not so good for AT-TPC data which can change density quite a bit. HDBSCAN however does not have this limitation. HDBSCAN scans density values forming clusters over different densities; the way it does this is fairly involved and we direct you to the [scikit-learn docs](https://scikit-learn.org/stable/modules/clustering.html#hdbscan) for more details.

We cluster on four features: the spatial coordinates (x, y, z) and the charge amplitude. Charge is included as this allows us to reject cross talk noise more effectively. However, for some datasets, it may be necessary to turn off the charge feature, particularly if the charge is very noisy. HDBSCAN, as implemented in scikit-learn, is also very fast, which is great because we have to do a lot of clustering!

However, this doesn't perfectly cluster our data right off the bat. HDBSCAN will make clusters over all the densities, but each density will get it's own cluster! This is not exactly what we want; we need to connect these pieces together to form the total trajectory.

The code for this is found in `spyral/core/clusterize.py` in the function `form_clusters`. There are several parameters to HDBSCAN that are exposed in the [configuration](../config/cluster.md) of Spyral.

## Joining of Clusters

The core method of joining is relatively simple: each initial cluster is fit with a circle in the X-Y plane. If the circles overlap significantly they are taken to be from the same trajectory. The required amount of overlap is a [configuration](../config/cluster.md) parameter to be tuned for each experiment. A secondary joining parameter is the relative mean charge of the two clusters to be joined; the two clusters should have a similar mean charge otherwise they do not belong together. Again, the threshold is a tunable parameter in the configuration.

Joining is repeated until clusters can no longer be joined together. These clusters are then taken to represent the complete detected particle trajectories for that event.

The code for this is found in `spyral/core/clusterize.py` in the functions `join_clusters` and `join_clusters_depth`.

## Cleanup of the Final Clusters

As a last stage, a final noise removal is applied to each cluster. The [Local Outlier Factor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html#sklearn.neighbors.LocalOutlierFactor) is an outlier test which compares the distance to nearest neighbors (neighbor density) to determine outlier points. The number of neighbors to use is an exposed [configuration](../config/cluster.md) parameter.

The code for this is found in `spyral/core/cluster.py`.

## Final Thoughts

The intial clustering and joining are two of the most important steps in Spyral, but they often require compromises. In our experience, there is no perfect set of parameters that will allow the clustering to work flawlessly for all events in a dataset; in fact given the inhomogenetiy of AT-TPC data, expecting one set of parameters to work all the time is somewhat crazy. In general, it is best to shoot for parameters that *generally* work. There will always be events that are too noisy or too fragmented for the clustering algorithm to work correctly. As such, this is the section of the code that most often requires fine tuning by hand.

In general, clustering is the second longest analysis phase, which makes sense given the complexity of much of the analysis. However, it is still typically fast compared to the point cloud phase.
