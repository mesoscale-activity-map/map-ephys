import datajoint as dj
from pipeline import lab, get_schema_name, foraging_analysis, report, psth_foraging, foraging_model
import multiprocessing as mp

# Ray does not support Windows, use multiprocessing instead
use_ray = False

def populatemytables_core(arguments,runround):
    if runround == 1:
        # foraging_analysis.TrialStats().populate(**arguments)
        # foraging_analysis.BlockStats().populate(**arguments)
        # foraging_analysis.SessionTaskProtocol().populate(**arguments)  #  Important for model fitting
        # foraging_analysis.SessionStats().populate(**arguments)
        # foraging_analysis.BlockFraction().populate(**arguments)
        # foraging_analysis.SessionMatching().populate(**arguments)
        # foraging_analysis.BlockEfficiency().populate(**arguments)           

        # report.SessionLevelForagingSummary.populate(**arguments)   
        # report.SessionLevelForagingLickingPSTH.populate(**arguments)   

        # psth_foraging.UnitPsth.populate(**arguments)
        
        foraging_model.FittedSessionModel.populate(**arguments)
        # psth_foraging.UnitPeriodLinearFit.populate(**arguments)
        
def populatemytables(paralel = True, cores = 9):
    IDs = {k: v for k, v in zip(*lab.WaterRestriction().fetch('water_restriction_number', 'subject_id'))}              
    if paralel:
        schema = dj.schema(get_schema_name('foraging_analysis'),locals())
        schema.jobs.delete()
    
        arguments = {'display_progress' : False, 'reserve_jobs' : True, 'order' : 'random'}
      
        for runround in [1]:
            print('round '+str(runround)+' of populate')
            
            result_ids = [pool.apply_async(populatemytables_core, args = (arguments,runround)) for coreidx in range(cores)] 
            
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
    
    cores = int(mp.cpu_count()) - 5  # Auto core number selection
    pool = mp.Pool(processes=cores)
    
    populatemytables(paralel=True, cores=cores)
    
    if pool != '':
        pool.close()
        pool.join()
