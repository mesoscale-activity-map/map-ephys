%{
# 3D trajectories of the tongue
-> tracking.Tracking
---
tongue_x_b    :   longblob
tongue_y_b    :   longblob
tongue_z_b    :   longblob
tongue_x_s    :   longblob
tongue_y_s    :   longblob
tongue_z_s    :   longblob
%}

classdef TongueTracking3D < dj.Computed
    properties
        keySource = (experiment.getSchema().v.Session & 'rig = "RRig-MTL"')...
            & (tracking.getSchema().v.Tracking & 'tracking_device = "Camera 3"')...
            & (tracking.getSchema().v.Tracking & 'tracking_device = "Camera 4"');
    end
    methods(Access=protected)        
        function make(self, key)
            
            [bot_tongue_x, bot_tongue_y] = fetchn(tracking.getSchema().v.TrackingTongueTracking & 'tracking_device = "Camera 4"' & key, 'tongue_x', 'tongue_y', 'ORDER BY trial');
            [sid_tongue_x, sid_tongue_y] = fetchn(tracking.getSchema().v.TrackingTongueTracking & 'tracking_device = "Camera 3"' & key, 'tongue_x', 'tongue_y', 'ORDER BY trial');
            
            if length(bot_tongue_x) ~= length(sid_tongue_x)
                disp('Mismatch in tracking bottom and side trials')
                return
            end
            
            load('Calib_Results_stereo.mat')
            
            key_insert=repmat(key,1,length(bot_tongue_x));
            
            for i = 1:length(bot_tongue_x) % loop over trials
                Bx=cell2mat(bot_tongue_x(i));
                By=cell2mat(bot_tongue_y(i));
                Sx=cell2mat(sid_tongue_x(i));
                Sy=cell2mat(sid_tongue_y(i));

                x_left_1=[Bx'; By']; x_right_1=[Sx'; Sy'];

                [Xc_1_left,Xc_1_right] = utils.stereo_triangulation(x_left_1,x_right_1,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
                key_insert(i).tongue_x_b = Xc_1_left(1,:);
                key_insert(i).tongue_y_b = Xc_1_left(2,:);
                key_insert(i).tongue_z_b = Xc_1_left(3,:);
                key_insert(i).tongue_x_s = Xc_1_right(1,:);
                key_insert(i).tongue_y_s = Xc_1_right(2,:);
                key_insert(i).tongue_z_s = Xc_1_right(3,:);
                key_insert(i).trial=i;
                
            end
            insert(self,key_insert);
        end
    end
end 