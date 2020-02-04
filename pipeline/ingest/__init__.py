import datajoint as dj


class ProbeInsertionError(Exception):
    """Raise when error encountered when ingesting probe insertion"""
    def __init__(self, msg=None):
        super().__init__('Probe Insertion Error: \n{}'.format(msg))
    pass


class ClusterMetricError(Exception):
    """Raise when error encountered when ingesting cluster metrics loaded from metrics.csv"""
    def __init__(self, msg=None):
        super().__init__('Cluster Metrics Error: \n{}'.format(msg))
    pass
