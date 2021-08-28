clear all

% Load configuration
dj.config();
dj.config.load('.\dj_local_conf.json')

global databasePrefix
databasePrefix = 'map_v2_';

schema_names = {'lab', 'experiment', 'histology', ...
    'ephys', 'tracking', 'ccf', 'oralfacial_analysis'};

for k = 1: numel(schema_names)
    clear schema
    schema_name = schema_names{k};
    eval(['import ', schema_name, '.getSchema']);
    eval(['schema = ', schema_name, '.getSchema();'])
    
    for j = 1:numel(schema.classNames)
        tbl_name = split(schema.classNames{j}, '.');
        tbl_name = join(tbl_name{2:end}, '.');
        eval(['v_', schema_name, '.(tbl_name) = schema.v.(tbl_name);']);
    end 
    
    eval(['v_', schema_name, '.schema = schema;'])
end

