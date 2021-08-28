%{
# the job reservation table for +tracking
table_name : varchar(255) # className of the table
key_hash   : char(32)     # key hash
-----
status    : enum("reserved","error","ignore") # if tuple is missing, the job is available
key=null           : blob                     # structure containing the key
error_message=""   : varchar(1023)            # error message returned if failed
error_stack=null   : blob                     # error stack if failed
user=""            : varchar(255)             # database user
host=""            : varchar(255)             # system hostname
pid=0              : int unsigned             # system process id
connection_id=0    : bigint unsigned          # database connection id
timestamp=CURRENT_TIMESTAMP : timestamp       # automatic timestamp
%}

classdef Jobs < dj.Jobs
end
