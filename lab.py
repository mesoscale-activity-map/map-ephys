import datajoint as dj

schema = dj.schema(dj.config['lab.database'])


@schema
class Animal(dj.Manual):
    definition = """
    animal  : int    # Janelia ANM ID (6 digits)
    ---
    dob    : date
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

@schema
class AnimalWaterRestriction(dj.Manual):
    # separated from Animal since is not an initial attribute;
    # potentially not applicable to all experiments
    definition = """
    -> Animal
    water_restriction : varchar(6) # water restriction number
    """

