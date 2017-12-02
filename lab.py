import datajoint as dj

schema = dj.schema(dj.config['lab.database'], locals())


@schema
class Animal(dj.Manual):
    definition = """
    animal  : int    # Janelia ANM ID (6 digits)
    ---
    dob    : date
	wr_num : varchar(6) # water restriction number 
    """


@schema
class Person(dj.Manual):
    definition = """
    username : varchar(12) 
    ----
    fullname : varchar(60)
    """

@schema
class Rig(dj.Manual):
    definition = """
    rig  : varchar(8)
    ---
    rig_description : varchar(1024) 
    """
