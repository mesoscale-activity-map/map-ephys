function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    global databasePrefix
    schemaObject = dj.Schema(dj.conn, 'histology', [databasePrefix, 'histology']);
end
obj = schemaObject;
end
