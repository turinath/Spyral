from .point_cloud import PointCloud
from .config import ClusterParameters
import numpy as np
from dataclasses import dataclass, field
from sklearn.neighbors import LocalOutlierFactor


@dataclass
class LabeledCloud:
    """Utility dataclass just for temporary holding in the clustering algorithims

    Attributes
    ----------
    label: int
        The label from the clustering algorithm
    point_cloud:
        The cluster data
    """

    label: int = -1  # default is noise label
    point_cloud: PointCloud = field(default_factory=PointCloud)
    clustered_data: np.ndarray = field(default_factory=lambda: np.zeros(0))


class Cluster:
    """Representation of trajectory cluster data.

    Attributes
    ----------
    event: int
        The event number
    label: int
        The cluster label from the algorithm
    data: ndarray
        The point cloud data (trimmed down). Contains position, integrated charge

    Methods
    -------
    Cluster(event: int=-1, label: int=-1, data: ndarray=np.empty(0, type=numpy.float64))
        Construct a Cluster
    from_labeled_cloud(cloud: LabeledCloud, params: ClusterParams)
        load the data from a LabeledCloud
    copy_cloud(cloud: PointCloud)
        copy the data from a PointCloud to the Cluster
    drop_outliers()
        Use the scikit-learn LocalOutlierFactor to identify and remove outliers in the Cluster
    """

    def __init__(
        self,
        event: int = -1,
        label: int = -1,
        data: np.ndarray = np.empty(0, dtype=np.float64),
    ):
        """Construct the Cluster

        Parameters
        ----------
        event: int
            The event number (default = -1)
        label: int
            The label from the clustering algorithm (default = -1)
        data: ndarray
            The PointCloud data for the Cluster (default = empty array)

        Returns
        -------
        Cluster
            An instance of the class
        """
        self.event = event
        self.label = label
        self.data = data

    def from_labeled_cloud(self, cloud: LabeledCloud, params: ClusterParameters):
        """Convert a LabeledCloud to a Cluster, dropping any outliers

        Parameters
        ----------
        cloud: LabeledCloud
            The LabeledCloud to convert
        params: ClusterParameters
            Configuration parameters for the cluster
        """
        self.event = cloud.point_cloud.event_number
        self.label = cloud.label
        self.copy_cloud(cloud.point_cloud)
        self.drop_outliers(params.outlier_scale_factor)

    def copy_cloud(self, cloud: PointCloud):
        """Copy PointCloud data to the cluster

        Copy a subset of the point cloud into the Cluster
        Only keep position, integrated charge

        Parameters
        ----------
        cloud: PointCloud
            The PointCloud to copy from
        """
        cloud.sort_in_z()
        self.data = np.zeros((len(cloud.cloud), 5))
        self.data[:, :3] = cloud.cloud[:, :3]  # position
        self.data[:, 3] = cloud.cloud[:, 4]  # peak integral
        self.data[:, 4] = cloud.cloud[:, 7]  # scale (big or small)

    def drop_outliers(self, scale: float = 0.05):
        """Use scikit-learn LocalOutlierFactor to test the cluster for spatial outliers.

        This helps reduce noise when fitting the data.

        Parameters
        ----------
        scale: float
            Scale factor to be multiplied by the length of the trajectory to get
            the number of neighbors over which to test
        """
        neighbors = int(scale * len(self.data))  # 0.05 default
        if neighbors < 2:
            neighbors = 2
        test_data = self.data[:, :3].copy()
        neigh = LocalOutlierFactor(n_neighbors=neighbors)
        result = neigh.fit_predict(test_data)
        self.data = self.data[result > 0]  # label=-1 is an outlier


def convert_labeled_to_cluster(
    cloud: LabeledCloud, params: ClusterParameters
) -> Cluster:
    """Function which takes in a LabeledCloud and ClusterParamters and returns a Cluster

    Parameters
    ----------
    cloud: LabeledCloud
        The LabeledCloud to convert
    params: ClusterParameters
        Configuration parameters for the cluster

    Returns
    -------
    Cluster
        The Cluster object
    """
    cluster = Cluster()
    cluster.from_labeled_cloud(cloud, params)
    return cluster
