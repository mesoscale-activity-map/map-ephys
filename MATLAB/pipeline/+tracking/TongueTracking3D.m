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


classdef TongueTracking3D < dj.Part
    properties
        keySource = (v_experiment.Session & 'rig = "RRig-MTL"') & (v_tracking.Tracking & 'tracking_device = "Camera 3"') & (v_tracking.Tracking & 'tracking_device = "Camera 4"');
    end
    methods(Access=protected)        
        function makeTuples(self, key)
            [bot_tongue_x, bot_tongue_y] = fetchn(v_tracking.TrackingTongueTracking & 'tracking_device = "Camera 4"' & key, 'tongue_x', 'tongue_y');
            [sid_tongue_x, sid_tongue_y] = fetchn(v_tracking.TrackingTongueTracking & 'tracking_device = "Camera 3"' & key, 'tongue_x', 'tongue_y');

            load('F:\3dCalib\Calib_Results_stereo.mat')

            Bx=cell2mat(bot_tongue_x(1));
            By=cell2mat(bot_tongue_y(1));
            Sx=cell2mat(sid_tongue_x(1));
            Sy=cell2mat(sid_tongue_y(1));

            x_left_1=[Bx'; By']; x_right_1=[Sx'; Sy'];

            [Xc_1_left,Xc_1_right] = stereo_triangulation(x_left_1,x_right_1,om,T,fc_left,cc_left,kc_left,alpha_c_left,fc_right,cc_right,kc_right,alpha_c_right);
            insert_key_tongue_position.tongue_x_b = Xc_1_left(1,:);
            insert_key_tongue_position.tongue_y_b = Xc_1_left(2,:);
            insert_key_tongue_position.tongue_z_b = Xc_1_left(3,:);
            insert_key_tongue_position.tongue_x_s = Xc_1_right(1,:);
            insert_key_tongue_position.tongue_y_s = Xc_1_right(2,:);
            insert_key_tongue_position.tongue_z_s = Xc_1_right(3,:);
            insert(self,insert_key_tongue_position);
            
        end
    end
end