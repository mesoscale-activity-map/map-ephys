function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    global databasePrefix
    schemaObject = dj.Schema(dj.conn, 'oralfacial_analysis', [databasePrefix, 'oralfacial_analysis']);
end
obj = schemaObject;
end
