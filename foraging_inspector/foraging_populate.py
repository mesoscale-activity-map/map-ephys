import datajoint as dj
from pipeline import pipeline_tools, lab, experiment, behavior_foraging
dj.conn()
try:
    import ray
    #%%
    @ray.remote
    def populatemytables_core_paralel(arguments,runround):
        if runround == 1:
            behavior_foraging.TrialReactionTime().populate(**arguments)
            behavior_foraging.BlockStats().populate(**arguments)
            behavior_foraging.SessionStats().populate(**arguments)
            behavior_foraging.SessionTaskProtocol().populate(**arguments)
            behavior_foraging.BlockRewardFractionNoBiasCheck().populate(**arguments)
            behavior_foraging.BlockChoiceFractionNoBiasCheck().populate(**arguments)
            behavior_foraging.SessionMatchBias().populate(**arguments)
            behavior_foraging.BlockEfficiency().populate(**arguments)   

            
except:
    pass
        
def populatemytables_core(arguments,runround):
    if runround == 1:
        behavior_foraging.TrialReactionTime().populate(**arguments)
        behavior_foraging.BlockStats().populate(**arguments)
        behavior_foraging.SessionStats().populate(**arguments)
        behavior_foraging.SessionTaskProtocol().populate(**arguments)
        behavior_foraging.BlockRewardFractionNoBiasCheck().populate(**arguments)
        behavior_foraging.BlockChoiceFractionNoBiasCheck().populate(**arguments)
        behavior_foraging.SessionMatchBias().populate(**arguments)
        behavior_foraging.BlockEfficiency().populate(**arguments) 
        
        
def populatemytables(paralel = True, cores = 9):
    IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}              
    if paralel:
        schema = dj.schema(pipeline_tools.get_schema_name('behavior_foraging'),locals())
        schema.jobs.delete()
        ray.init(num_cpus = cores)
        for runround in [1]:
            arguments = {'display_progress' : False, 'reserve_jobs' : True,'order' : 'random'}
            print('round '+str(runround)+' of populate')
            result_ids = []
            for coreidx in range(cores):
                result_ids.append(populatemytables_core_paralel.remote(arguments,runround))        
            ray.get(result_ids)
            arguments = {'display_progress' : True, 'reserve_jobs' : False}
            populatemytables_core(arguments,runround)            
        ray.shutdown()
    else:
        for runround in [1]:
            arguments = {'display_progress' : True, 'reserve_jobs' : False,'order' : 'random'}
            populatemytables_core(arguments,runround)