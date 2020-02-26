import datajoint as dj
import pymysql.err


def keep_conn_alive():
    try:
        dj.conn().ping()
    except pymysql.err.Error:
        dj.conn().connect()


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


class BitCodeError(Exception):
    """Raise when error encountered when extracting information from bitcode file"""
    def __init__(self, msg=None):
        super().__init__('BitCode Error: \n{}'.format(msg))
    pass


class IdenticalClusterResultError(Exception):
    """Raise when identical clustering time found between the existing clustering results and the replacement one"""
    def __init__(self, identical_clustering_results, msg=''):
        emsg = ['\treplacement clustering dir ({}) for probe insertion {}\n'.format(d, n)
                for n, d in identical_clustering_results]
        super().__init__('Identical clustering time found for:\n{}{}'.format(''.join(emsg), msg))
    pass
