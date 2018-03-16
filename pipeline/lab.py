import datajoint as dj

schema = dj.schema(dj.config['lab.database'])


@schema
class Person(dj.Manual):
    definition = """
    username : varchar(24) 
    ----
    fullname : varchar(255)
    """


@schema
class Rig(dj.Manual):
    definition = """
    rig             : varchar(24)
    ---
    room            : varchar(20) # example 2w.342
    rig_description : varchar(1024) 
    """


@schema
class AnimalSource(dj.Lookup):
    definition = """
    animal_source       : varchar(30)
    """
    contents = zip(['Jackson Labs', 'Allen Institute', 'Charles River', 'MMRRC', 'Taconic', 'Other'])


@schema
class ModifiedGene(dj.Manual):
    definition = """
    gene_modification   : varchar(60)
    ---
    description = ''         : varchar(256)
    """


@schema
class Subject(dj.Manual):
    definition = """
    subject_id          : int   # institution 6 digit animal ID
    ---
    -> [nullable] Person        # person responsible for the animal
    cage_number         : int   # institution 6 digit animal ID
    date_of_birth       : date  # format: yyyy-mm-dd
    sex                 : enum('M','F','Unknown')
    -> [nullable] AnimalSource  # where was the animal ordered from
    """

    class GeneModification(dj.Part):
        definition = """
        # Subject gene modifications
        -> Subject
        -> ModifiedGene
        ---
        zygosity = 'Unknown' : enum('Het', 'Hom', 'Unknown')
        type = 'Unknown'     : enum('Knock-in', 'Transgene', 'Unknown')
        """


@schema
class CompleteGenotype(dj.Computed):
    # should be computed
    definition = """
    -> Subject
    ---
    complete_genotype : varchar(1000)
    """

    def make(self, key):
        pass


@schema
class WaterRestriction(dj.Manual):
    definition = """
    -> Subject
    ---
    water_restriction_number    : varchar(16)   # WR number
    cage_number                 : int
    wr_start_date               : date
    wr_start_weight             : Decimal(6,3)
    """


@schema
class VirusSource(dj.Lookup):
    definition = """
    virus_source   : varchar(60)
    """
    contents = zip(['Janelia', 'UPenn', 'Addgene', 'UNC', 'Other'])


@schema
class Serotype(dj.Manual):
    definition = """
    serotype   : varchar(60)
    """


@schema
class Virus(dj.Manual):
    definition = """
    virus_id : int unsigned
    ---
    -> VirusSource 
    -> Serotype
    -> Person
    virus_name      : varchar(256)
    titer           : Decimal(20,1) # 
    order_date      : date
    remarks         : varchar(256)
    """

    class Notes(dj.Part):
        definition = """
        # Notes for virus
        -> Virus
        note_id     : int
        ---
        note        : varchar(256)
        """


@schema
class SkullReference(dj.Lookup):
    definition = """
    skull_reference   : varchar(60)
    """
    contents = zip(['Bregma', 'Lambda'])

    
@schema
class BrainArea(dj.Lookup):
    definition = """
    brain_area = 'ALM'  : varchar(32)
    ---
    description = null : varchar (4000) # name of the brain area
    """
    contents = [('ALM', 'anterior lateral motor cortex'), ('vS1', 'vibrissal primary somatosensory cortex ("barrel cortex")')]
    
    
@schema
class Hemisphere(dj.Lookup):
    definition = """
    hemisphere = 'left'   : varchar(32)
    """
    contents = zip(['left', 'right'])
    
    
@schema
class Surgery(dj.Manual):
    definition = """
    -> Subject
    surgery_id          : int      # surgery number
    ---
    -> Person
    start_time          : datetime # start time
    end_time            : datetime # end time
    description         : varchar(256)
    """

    class VirusInjection(dj.Part):
        definition = """
        # Virus injections
        -> Surgery
        injection_id : int
        ---
        -> Virus
        -> SkullReference
        ml_location     : Decimal(8,3) # um from ref left is positive
        ap_location     : Decimal(8,3) # um from ref anterior is positive
        dv_location     : Decimal(8,3) # um from dura dorsal is positive 
        volume          : Decimal(10,3) # in nl
        dilution        : Decimal (10, 2) # 1 to how much
        description     : varchar(256)
        """

    class Procedure(dj.Part):
        definition = """
        # Other things you did to the animal
        -> Surgery
        procedure_id : int
        ---
        -> SkullReference
        ml_location=null     : Decimal(8,3) # um from ref left is positive
        ap_location=null     : Decimal(8,3) # um from ref anterior is positive
        dv_location=null     : Decimal(8,3) # um from dura dorsal is positive 
        description     : varchar(1000)
        """


@schema
class SurgeryLocation(dj.Manual):
    definition = """
    -> Surgery.Procedure
    ---
    -> Hemisphere
    -> BrainArea 
    """