function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    schemaObject = dj.Schema(dj.conn, 'oralfacial_analysis', 'daveliu_analysis');
end
obj = schemaObject;
end
