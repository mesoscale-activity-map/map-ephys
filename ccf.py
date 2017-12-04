import datajoint as dj


schema = dj.schema(dj.config['ccf.database'], locals())


@schema
class CCF(dj.Lookup):
    definition = """
    # Common Coordinate Framework
    x   :  int   # (um)
    y   :  int   # (um)
    z   :  int   # (um)
    ---
    label  : varchar(128)   # TODO
    """

@schema
class AnnotationType(dj.Lookup):
    definition = """
    annotation_type  : varchar(16)  
    """
    
@schema
class CCFAnnotation(dj.Manual):
    definition = """
    -> CCF
    -> AnnotationType
    ---
    annotation  : varchar(1200)
    """
