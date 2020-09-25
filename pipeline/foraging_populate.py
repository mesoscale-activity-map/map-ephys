import datajoint as dj
from pipeline import lab, get_schema_name, foraging_analysis

try:
    import ray
    #%%
    @ray.remote
    def populatemytables_core_paralel(arguments,runround):
        if runround == 1:
            foraging_analysis.TrialStats().populate(**arguments)
            foraging_analysis.BlockStats().populate(**arguments)
            foraging_analysis.SessionStats().populate(**arguments)
            foraging_analysis.SessionTaskProtocol().populate(**arguments)
            foraging_analysis.BlockFraction().populate(**arguments)
            foraging_analysis.SessionMatching().populate(**arguments)
            foraging_analysis.BlockEfficiency().populate(**arguments)   
            
    use_ray = True
except:
    # Ray does not support Windows, use multiprocessing instead
    use_ray = False
    import multiprocessing as mp
    def populatemytables_core_paralel(arguments,runround):
        if runround == 1:
            foraging_analysis.TrialStats().populate(**arguments)
            foraging_analysis.BlockStats().populate(**arguments)
            foraging_analysis.SessionStats().populate(**arguments)
            foraging_analysis.SessionTaskProtocol().populate(**arguments)
            foraging_analysis.BlockFraction().populate(**arguments)
            foraging_analysis.SessionMatching().populate(**arguments)
            foraging_analysis.BlockEfficiency().populate(**arguments)   
    
def populatemytables_core(arguments,runround):
    if runround == 1:
        foraging_analysis.TrialStats().populate(**arguments)
        foraging_analysis.BlockStats().populate(**arguments)
        foraging_analysis.SessionStats().populate(**arguments)
        foraging_analysis.SessionTaskProtocol().populate(**arguments)
        foraging_analysis.BlockFraction().populate(**arguments)
        foraging_analysis.SessionMatching().populate(**arguments)
        foraging_analysis.BlockEfficiency().populate(**arguments)           
        
def populatemytables(paralel = True, cores = 9):
    IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}              
    if paralel:
        schema = dj.schema(get_schema_name('foraging_analysis'),locals())
        schema.jobs.delete()
        
        if use_ray:
            ray.init(num_cpus = cores)
            arguments = {'display_progress' : False, 'reserve_jobs' : True,'order' : 'random'}
            
            for runround in [1]:
                print('round '+str(runround)+' of populate')
                result_ids = []
                for coreidx in range(cores):
                    result_ids.append(populatemytables_core_paralel.remote(arguments,runround))        
                ray.get(result_ids)
            ray.shutdown()
        else:  # Use multiprocessing
            arguments = {'display_progress' : False, 'reserve_jobs' : True, 'order' : 'random'}
          
            for runround in [1]:
                print('round '+str(runround)+' of populate')
                
                result_ids = [pool.apply_async(populatemytables_core_paralel, args = (arguments,runround)) for coreidx in range(cores)] 
                
                for result_id in result_ids:
                    result_id.get()

        
        # Just in case there're anything missing?
        arguments = {'display_progress' : True, 'reserve_jobs' : False}
        populatemytables_core(arguments,runround)            
            
    else:
        for runround in [1]:
            arguments = {'display_progress' : True, 'reserve_jobs' : False,'order' : 'random'}
            populatemytables_core(arguments,runround)
            
            
if __name__ == '__main__' and use_ray == False:  # This is a workaround for mp.apply_async to run in Windows

    # from pipeline import shell
    # shell.logsetup('INFO')
    # shell.ingest_foraging_behavior()
    
    cores = int(mp.cpu_count()) - 2  # Auto core number selection
    pool = mp.Pool(processes=cores)
    
    populatemytables(paralel=True, cores=cores)
    
    if pool != '':
        pool.close()
        pool.join()
