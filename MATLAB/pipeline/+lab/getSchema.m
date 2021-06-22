function obj = getSchema
persistent schemaObject
if isempty(schemaObject)
    global databasePrefix
    schemaObject = dj.Schema(dj.conn, 'lab', [databasePrefix, 'lab']);
end
obj = schemaObject;
end
