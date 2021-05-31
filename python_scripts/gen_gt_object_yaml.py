import numpy as np 
import os, sys 
import yaml

from scipy.spatial.transform import Rotation as R

# input format 
# x y z roll pitch yaw (in degrees)

# use squaternion to convert to Quaternion
# euler angles from_eluer(roll, pitch, yaw), default is radians, but set
# degrees true if giving degrees
# q = Quaternion.from_euler(0, -90, 100, degrees=True)

if __name__ == '__main__':

    root_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '')) 

    input_launch_file = root_dir + '/cache/overpasscity_with_objects.yaml'
    output_yaml_file = root_dir + '/cache/objects_gt.yaml'

    lines = open(input_launch_file, "r").readlines()

    key_word1 = 'spawn'
    key_word2 = 'Spawner'

    objects_ids_gt = []
    objects_class_gt = []
    objects_rotation_gt = []
    objects_translation_gt = []
    total_object_num = 0

    for line in lines:
        
        if (key_word1 not in line or key_word2 not in line):
            continue 

        words = line.split()
        
        # get object id 
        object_id = int(words[4].split("_")[-1])
        objects_ids_gt.append(object_id)

        object_class = words[4].split("_")[0]
        objects_class_gt.append(object_class)

        # get orientation 
        roll, pitch, yaw = np.double(words[8]), np.double(words[9]), np.double(words[10].split("'")[0])
        
        # r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        # unity frame may define yaw the opposite direction 
        r = R.from_euler('xyz', [roll, pitch, -yaw], degrees=True)

        # convert to quaternion 
        # q = r.as_quat()

        rot_mat = r.as_dcm().tolist()
        for row in rot_mat:
            objects_rotation_gt += row

        # get position 
        x, y, z = np.float(words[5]), np.float(words[6]), np.float(words[7])

        position = [x, y, z]
        objects_translation_gt += position

        total_object_num += 1

    with open(output_yaml_file, 'w') as f:

        output_dict = {}
        output_dict['total_object_num'] = total_object_num
        data = yaml.safe_dump(output_dict, f, default_style=None, default_flow_style=True)

        output_dict = {}
        output_dict['objects_ids_gt'] = objects_ids_gt
        data = yaml.safe_dump(output_dict, f, default_style=None, default_flow_style=True)

        output_dict = {}
        output_dict['objects_class_gt'] = objects_class_gt
        data = yaml.safe_dump(output_dict, f, default_style=None, default_flow_style=True)

        output_dict = {}
        output_dict['objects_rotation_gt'] = objects_rotation_gt
        data = yaml.safe_dump(output_dict, f, default_style=None, default_flow_style=True)
        
        output_dict = {}
        output_dict['objects_translation_gt'] = objects_translation_gt
        data = yaml.safe_dump(output_dict, f, default_style=None, default_flow_style=True)
