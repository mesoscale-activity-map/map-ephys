function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    global databasePrefix
    schemaObject = dj.Schema(dj.conn, 'ccf', [databasePrefix, 'ccf']);
end
obj = schemaObject;
end
