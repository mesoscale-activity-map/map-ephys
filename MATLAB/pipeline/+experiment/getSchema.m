function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    global databasePrefix
    schemaObject = dj.Schema(dj.conn, 'experiment', [databasePrefix, 'experiment']);
end
obj = schemaObject;
end
