function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    global databasePrefix
    schemaObject = dj.Schema(dj.conn, 'tracking', [databasePrefix, 'tracking']);
end
obj = schemaObject;
end
