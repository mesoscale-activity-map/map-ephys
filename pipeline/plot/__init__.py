
from .util import show_source

# from .unit_psth import unit_psth
# from .unit_psth import unit_psth_ll
#
# from .group_psth import group_psth
# from .group_psth import group_psth_ll
#
# __all__ = [show_source,
#            unit_psth, unit_psth_ll,
#            group_psth, group_psth_ll]

__all__ = [show_source]


class PhotostimError(Exception):
    """Raise when error encountered for photostim-related plots"""
    def __init__(self, msg=None):
        super().__init__('PhotostimError: \n{}'.format(msg))
    pass
