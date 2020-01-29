import datajoint as dj
import numpy as np
from . import get_schema_name

schema = dj.schema(get_schema_name('lab'))


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
class AnimalStrain(dj.Lookup):
    definition = """
    animal_strain       : varchar(30)
    """
    contents = zip(['pl56', 'kj18'])


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
    gene_modification_description = ''         : varchar(256)
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

    class Strain(dj.Part):
        definition = """
        # Subject strains
        -> master
        -> AnimalStrain
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
    brain_area: varchar(32)
    ---
    description = null : varchar (4000) # name of the brain area (lab terms, not necessarily in AIBS)
    """
    contents = [('ALM', 'anterior lateral motor cortex'),
                ('vS1', 'vibrissal primary somatosensory cortex ("barrel cortex")'),
                ('Thalamus', 'Thalamus'), ('Medulla', 'Medulla'),
                ('Striatum', 'Striatum'), ('Midbrain', 'Midbrain')]
    
    
@schema
class Hemisphere(dj.Lookup):
    definition = """
    hemisphere: varchar(32)
    """
    contents = zip(['left', 'right', 'both'])


@schema
class Surgery(dj.Manual):
    definition = """
    -> Subject
    surgery_id          : int      # surgery number
    ---
    -> Person
    start_time          : datetime # start time
    end_time            : datetime # end time
    surgery_description : varchar(256)
    """

    class VirusInjection(dj.Part):
        definition = """
        # Virus injections
        -> master
        injection_id : int
        ---
        -> Virus
        -> SkullReference
        ap_location     : Decimal(8,3) # um from ref anterior is positive
        ml_location     : Decimal(8,3) # um from ref right is positive 
        dv_location     : Decimal(8,3) # um from dura dorsal is positive 
        volume          : Decimal(10,3) # in nl
        dilution        : Decimal (10, 2) # 1 to how much
        description     : varchar(256)
        """

    class Procedure(dj.Part):
        definition = """
        # Other things you did to the animal
        -> master
        procedure_id : int
        ---
        -> SkullReference
        ap_location=null     : Decimal(8,3) # um from ref anterior is positive
        ml_location=null     : Decimal(8,3) # um from ref right is positive
        dv_location=null     : Decimal(8,3) # um from dura dorsal is positive 
        surgery_procedure_description     : varchar(1000)
        """


@schema
class SurgeryLocation(dj.Manual):
    definition = """
    -> Surgery.Procedure
    ---
    -> Hemisphere
    -> BrainArea 
    """


@schema
class ProbeType(dj.Lookup):
    definition = """
    probe_type: varchar(32)  # e.g. neuropixels_1.0 
    """

    class Electrode(dj.Part):
        definition = """
        -> master
        electrode: int       # electrode index, starts at 0
        ---
        shank: int           # shank index, starts at 0, advance left to right
        shank_col: int       # column index, starts at 0, advance left to right
        shank_row: int       # row index, starts at 0, advance tip to tail
        x_coord=NULL: float  # (um) x coordinate of the electrode within the probe, (0, 0) is the tip of the probe
        y_coord=NULL: float  # (um) y coordinate of the electrode within the probe, (0, 0) is the tip of the probe
        z_coord=0: float     # (um) z coordinate of the electrode within the probe, (0, 0) is the tip of the probe
        """

    @property
    def contents(self):
        return zip(['silicon_probe', 'tetrode_array',
                    'neuropixels 1.0 - 3A', 'neuropixels 1.0 - 3B',
                    'neuropixels 2.0 - SS', 'neuropixels 2.0 - MS'])

    @staticmethod
    def create_neuropixels_probe(probe_type='neuropixels 1.0 - 3A'):
        """
        Create `ProbeType` and `Electrode` for neuropixels probe 1.0 (3A and 3B), 2.0 (SS and MS)
        For electrode location, the (0, 0) is the bottom left corner of the probe (ignore the tip portion)
        Electrode numbering is 1-indexing
        """

        def build_electrodes(site_count, col_spacing, row_spacing, white_spacing, col_count=2,
                             shank_count=1, shank_spacing=250):
            """
            :param site_count: site count per shank
            :param col_spacing: (um) horrizontal spacing between sites
            :param row_spacing: (um) vertical spacing between columns
            :param white_spacing: (um) offset spacing
            :param col_count: number of column per shank
            :param shank_count: number of shank
            :param shank_spacing: spacing between shanks
            :return:
            """
            row_count = int(site_count / col_count)
            x_coords = np.tile([0, 0 + col_spacing], row_count)
            x_white_spaces = np.tile([white_spacing, white_spacing, 0, 0], int(row_count / 2))

            x_coords = x_coords + x_white_spaces
            y_coords = np.repeat(np.arange(row_count) * row_spacing, 2)

            shank_cols = np.tile([0, 1], row_count)
            shank_rows = np.repeat(range(row_count), 2)

            npx_electrodes = []
            for shank_no in range(shank_count):
                npx_electrodes.extend([{'electrode': (site_count * shank_no) + e_id + 1,  # electrode number is 1-based index
                                        'shank': shank_no,
                                        'shank_col': c_id,
                                        'shank_row': r_id,
                                        'x_coord': x + (shank_no * shank_spacing),
                                        'y_coord': y,
                                        'z_coord': 0} for e_id, (c_id, r_id, x, y) in enumerate(
                    zip(shank_cols, shank_rows, x_coords, y_coords))])

            return npx_electrodes

        # ---- 1.0 3A ----
        if probe_type == 'neuropixels 1.0 - 3A':
            electrodes = build_electrodes(site_count = 960, col_spacing = 32, row_spacing = 20,
                                          white_spacing = 16, col_count = 2)

            probe_type = {'probe_type': 'neuropixels 1.0 - 3A'}
            with ProbeType.connection.transaction:
                ProbeType.insert1(probe_type, skip_duplicates=True)
                ProbeType.Electrode.insert([{**probe_type, **e} for e in electrodes], skip_duplicates=True)

        # ---- 1.0 3B ----
        if probe_type == 'neuropixels 1.0 - 3B':
            electrodes = build_electrodes(site_count = 960, col_spacing = 32, row_spacing = 20,
                                          white_spacing = 16, col_count = 2)

            probe_type = {'probe_type': 'neuropixels 1.0 - 3B'}
            with ProbeType.connection.transaction:
                ProbeType.insert1(probe_type, skip_duplicates=True)
                ProbeType.Electrode.insert([{**probe_type, **e} for e in electrodes], skip_duplicates=True)

        # ---- 2.0 Single shank ----
        if probe_type == 'neuropixels 2.0 - SS':
            electrodes = build_electrodes(site_count=1280, col_spacing=32, row_spacing=15,
                                          white_spacing=0, col_count=2,
                                          shank_count=1, shank_spacing=250)

            probe_type = {'probe_type': 'neuropixels 2.0 - SS'}
            with ProbeType.connection.transaction:
                ProbeType.insert1(probe_type, skip_duplicates=True)
                ProbeType.Electrode.insert([{**probe_type, **e} for e in electrodes], skip_duplicates=True)

        # ---- 2.0 Multi shank ----
        if probe_type == 'neuropixels 2.0 - MS':
            electrodes = build_electrodes(site_count=1280, col_spacing=32, row_spacing=15,
                                          white_spacing=0, col_count=2,
                                          shank_count=4, shank_spacing=250)

            probe_type = {'probe_type': 'neuropixels 2.0 - MS'}
            with ProbeType.connection.transaction:
                ProbeType.insert1(probe_type, skip_duplicates=True)
                ProbeType.Electrode.insert([{**probe_type, **e} for e in electrodes], skip_duplicates=True)


@schema
class ElectrodeConfig(dj.Lookup):
    definition = """
    -> ProbeType
    electrode_config_name: varchar(64)  # user friendly name
    ---
    electrode_config_hash: varchar(36)  # hash of the group and group_member (ensure uniqueness)
    unique index (electrode_config_hash)
    """

    class ElectrodeGroup(dj.Part):
        definition = """
        # grouping of electrodes to be clustered together (e.g. a neuropixel electrode config - 384/960)
        -> master
        electrode_group: int  # electrode group
        """

    class Electrode(dj.Part):
        definition = """
        -> master.ElectrodeGroup
        -> ProbeType.Electrode
        ---
        is_used: bool  # is this channel used for spatial average (ref channels are by default not used)
        """


@schema
class Probe(dj.Lookup):
    definition = """  # represent a physical probe
    probe: varchar(32)  # unique identifier for this model of probe (e.g. part number)
    ---
    -> ProbeType
    probe_comment='' :  varchar(1000)
    """


@schema
class PhotostimDevice(dj.Lookup):
    definition = """
    photostim_device  : varchar(20)
    ---
    excitation_wavelength :  decimal(5,1)  # (nm) 
    photostim_device_description : varchar(255)
    """
    contents =[
       ('LaserGem473', 473, 'Laser (Laser Quantum, Gem 473)'),
       ('LED470', 470, 'LED (Thor Labs, M470F3 - 470 nm, 17.2 mW (Min) Fiber-Coupled LED)'),
       ('OBIS470', 473, 'OBIS 473nm LX 50mW Laser System: Fiber Pigtail (Coherent Inc)')]
