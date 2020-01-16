import datajoint as dj


class ProbeInsertionError(Exception):
    """Raise when error encountered when ingesting probe insertion"""
    def __init__(self, msg=None):
        super().__init__('Probe Insertion Error: \n{}'.format(msg))
    pass
