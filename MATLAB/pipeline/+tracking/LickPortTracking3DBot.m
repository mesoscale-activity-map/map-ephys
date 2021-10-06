%{
# 3D trajectories of the lickport
-> tracking.Tracking
---
lickport_x    :   longblob
lickport_y    :   longblob
lickport_z    :   longblob
%}

classdef LickPortTracking3DBot < dj.Computed
    properties
        keySource = (experiment.getSchema().v.Session & 'rig = "RRig-MTL"')...
            & (tracking.getSchema().v.Tracking & 'tracking_device = "Camera 3"')...
            & (tracking.getSchema().v.Tracking & 'tracking_device = "Camera 4"');
    end
    methods(Access=protected)        
        function makeTuples(self, key)            
            [bot_lickport_x, bot_lickport_y] = fetchn(tracking.getSchema().v.TrackingLickPortTracking & 'tracking_device = "Camera 4"' & key, 'lickport_x', 'lickport_y', 'ORDER BY trial');
            [sid_lickport_x, sid_lickport_y] = fetchn(tracking.getSchema().v.TrackingLickPortTracking & 'tracking_device = "Camera 3"' & key, 'lickport_x', 'lickport_y', 'ORDER BY trial');
            
            if length(bot_lickport_x) ~= length(sid_lickport_x)
                disp('Mismatch in tracking bottom and side trials')
                return
            end
            
            load('Calib_Results_stereo.mat')
            key.tracking_device='Camera 4';
            key_insert=repmat(key,1,length(bot_lickport_x));
            
            counter=0;
            for i = 1:length(bot_lickport_x) % loop over trials
                x_left_1=[cell2mat(bot_lickport_x(i))'; cell2mat(bot_lickport_y(i))']; x_right_1=[cell2mat(sid_lickport_x(i))'; cell2mat(sid_lickport_y(i))'];
                
                if size(x_left_1,2)==size(x_right_1,2)
                    [Xc_1_left,~] = utils.stereo_triangulation(x_left_1,x_right_1,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
                    key_insert(i-counter).lickport_x = Xc_1_left(1,:);
                    key_insert(i-counter).lickport_y = Xc_1_left(2,:);
                    key_insert(i-counter).lickport_z = Xc_1_left(3,:);
                    key_insert(i-counter).trial=i;
                else
                    key_insert=key_insert([1:i-counter-1 i-counter+1:end]);
                    counter=counter+1;
                end
            end
            insert(self,key_insert);
        end
    end
end 