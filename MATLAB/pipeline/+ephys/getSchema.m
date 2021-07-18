function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    global databasePrefix
    schemaObject = dj.Schema(dj.conn, 'ephys', [databasePrefix, 'ephys']);
end
obj = schemaObject;
end
