# Clustering Configuration

The Cluster parameters which control the clustering, joining, and outlier detection algorithms. The default cluster parameters given in `config.json` are:

```json
"Cluster":
{
    "min_cloud_size": 50,
    "smoothing_neighbor_distance(mm)": 15.0,
    "minimum_points": 5,
    "minimum_size_scale_factor": 0.05,
    "minimum_size_lower_cutoff": 10,
    "cluster_selection_epsilon": 0.3,
    "circle_overlap_ratio": 0.50,
    "fractional_charge_threshold": 0.7,
    "outlier_scale_factor": 0.05
},
```

A break down of each parameter:

## min_cloud_size

This is the minimum size a point cloud must be (in number of points) to be sent through the clustering algorithm. Any smaller point clouds will be tossed as noise.

## smoothing_neighbor_distance(mm)

The neighborhood search radius in millimeters for the smoothing algorithm. Before data is clustered, it is smoothed by averaging over neighbors. This value should be small to avoid smoothing artifacts. This is a scale-dependent parameter and will need adjusted for each experiment.

## minimum_points

The minimum number of samples (points) in a neighborhood for a point to be a core point. This is a re-exposure of the `min_samples` parameter of [scikit-learn's HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN). See their documentation for more details. Larger values will make the algorithm more likely to identify points as noise. See the original [HDBSCAN docs](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#) for details on why this parameter is important and how it can impact the data.

## minimum_size_scale_factor

HDBSCAN requires a minimum size (the hyper parameter `min_cluster_size` in [scikit-learn's HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN)) in terms of samples for a group to be considered a valid cluster. AT-TPC point clouds vary dramatically in size, from tens of points to thousands. To handle this wide scale, we use a scale factor to determine the appropriate minimum size, where `min_cluster_size = minimum_size_scale_factor * n_cloud_points`. The default value was found through some testing, and may need serious adjustment to produce best results. Note that the scale factor should be *small*.

## minimum_size_lower_cutoff

As discussed in the above `minimum_size_scale_factor`, we need to scale the `min_cluster_size` parameter to the size of the point cloud. However, there must be a lower limit (i.e. you can't have a minimum cluster size of 0). This parameter sets the lower limit; that is any `min_cluster_size` calculated using the scale factor that is smaller than this cutoff is replaced with the cutoff value. As an example, if the cutoff is set to 10 and the calculated value is 50, the calculated value would be used. However, if the calculated value is 5, the cutoff would be used instead.

## cluster_selection_epsilon

A re-exposure of the `cluster_selection_epsilon` paramter of [scikit-learn's HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN). This parameter will merge clusters that are less than epsilon apart. Note that this epsilon must be on the scale of the scaled data (i.e. it is not in normal units). The impact of this parameter is large, and small changes to this value can produce dramatically different results. Larger values will bias the clustering to assume the point cloud is onesingle cluster (or all noise), while smaller values will cause the algorithm to revert to the default result of HDBSCAN. See the original [HDBSCAN docs](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#) for details on why this parameter is important and how it can impact the data.

## circle_overlap_ratio

The minimum amount of overlap between circles fit to two clusters for the clusters to be joined together into a single cluster. HDBSCAN often fractures trajectories into multiple clusters as the point density changes due to the pad size, gaps, etc. These fragments are grouped together based on how much circles fit on their 2-D projections (X-Y) overlap.

## fractional_charge_threshold

The maximum allowed difference between the mean charge of two clusters (relative to the larger of the two means) for the clusters to be joined together into a single cluster. HDBSCAN often fractures trajectories into multiple clusters as the point density changes due to the pad size, gaps, etc. These fragments are grouped together based on how similar their mean charges are.

## outlier_scale_factor

We use [scikit-learn's LocalOutlierFactor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html) as a last round of noise elimination on a cluster-by-cluster basis. This algorithim requires a number of neighbors to search over (the `n_neighbors` parameter). As with the `min_cluster_size` in HDBSCAN, we need to scale this value off the size of the cluster. This factor multiplied by the size of the cluster gives the number of neighbors to search over (`n_neighbors = outlier_scale_factor * cluster_size`). This value tends to have a "sweet spot" where it is most effective. If it is too large, every point has basically the same outlier factor as you're including the entire cluster for every point. If it is too small the variance between neighbors can be too large and the results will be unpredictable. Note that if the value of `outlier_scale_factor * cluster_size` is less than 2, `n_neighbors` will be set to 2 as this is the minimum allowed value.
