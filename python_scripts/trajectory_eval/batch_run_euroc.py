import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))+"/../")

import path_def 

root_dir = path_def.orcvio_root_path

launch_file = root_dir + '/ros_wrapper/src/orcvio/launch/euroc/orcvio_euroc.launch'
bag_dir = path_def.euroc_dataset_path

# bag_name:bag_start in sec
bag_list = {'MH_01_easy':0,
            'MH_02_easy':0,
            'MH_03_medium':0,
            'MH_04_difficult':0,
            'MH_05_difficult':3,
            'V1_01_easy':5,
            'V1_02_medium':10,
            'V1_03_difficult':6,
            'V2_01_easy':4,
            'V2_02_medium':4,
            'V2_03_difficult':5}

# bag_list = {'MH_01_easy':0}
run_cmd = 'roslaunch orcvio orcvio_euroc.launch'
run_res = path_def.euroc_cache_path + 'temp_rmse.txt'
res_table = os.path.join(path_def.euroc_cache_path, 'orcvio_result.md')
png_dir = os.path.join(path_def.euroc_cache_path, 'figure')

marker_table = ['o','*','^','s','p','+','x','d','h','v','<','>','1','2','3']
png_idx = 0

def run_once():
    # return tuple((np.random.rand(2)*3).tolist())
    if os.path.exists(run_res):
        os.remove(run_res)
    os.system(run_cmd)
    if os.path.exists(run_res):
        return [float(x) for x in open(run_res).readlines()[0].split(' ')]
    else:
        return [-1,-1]

# [name:(type,value)]
def modify_launch(params):
    lines = open(launch_file, "r").readlines()
    fp = open(launch_file, "w")
    for line in lines:
        for name in params.keys():
            if name in line and ("rosbag" not in line):
                a, b = params[name]
                if (name == 'path_bag'):
                    line = '    <arg name="%s" default="%s" />\n'%(name,b)
                else: 
                    line = '    <param name="%s" type="%s" value="%s" />\n'%(name,a,b)
                break
        fp.write(line)
    fp.close()

def average_rmse(rmse):
    deg, meter = np.mean([(a,b) for a,b in rmse if a>0 and b>0], axis=0)
    return deg, meter

def plot_rmse(res_list, save_png):
    fig = plt.figure(figsize=(16, 8))
    names = [s[:5] for s in bag_list.keys()] + ['avg']
    plt.subplot(121)
    for i, res in enumerate(res_list):
        lege, rmse = res
        data = np.array(rmse)
        plt.plot(data[:,0], marker_table[i]+'-', label=lege)
    plt.legend()
    plt.xticks(range(len(names)),names,rotation=60)
    plt.ylabel('orientation error[degree]')
    plt.ylim([0,5])
    plt.subplot(122)
    for i, res in enumerate(res_list):
        lege, rmse = res
        data = np.array(rmse)
        plt.plot(data[:,1], marker_table[i]+'-', label=lege)
    plt.legend()
    plt.xticks(range(len(names)),names,rotation=60)
    plt.ylabel('position error[m]')
    plt.ylim([0,1])
    plt.savefig(save_png)

def loop_rosbag(params):
    rmse = []
    params = copy.deepcopy(params)
    for bag, start_ts in bag_list.items():
        fbag = os.path.join(bag_dir, bag + '.bag')
        fcsv = "$(find orcvio)/../../../eval/euroc_mav/%s.csv"%bag
        if os.path.exists(fbag):
            params['path_bag'] = ('string', fbag)
            params['path_gt'] = ('string', fcsv)
            modify_launch(params)
            res = run_once()
            rmse.append(res)
        else:
            rmse.append((-1,-1))
    rmse.append(average_rmse(rmse))
    return rmse

def batch_run_single_change(param_table, default_params, fp):
    default_res = loop_rosbag(default_params)
    for param_name, param_type, choice in param_table:
        fp.write('|%s|%s|avg|\n'%(param_name, '|'.join([s[:5] for s in bag_list.keys()])))
        fp.write('|'+'--|'*(len(bag_list.keys())+2)+'\n')
        fp.write('|%s|%s|\n'%(choice[0],'|'.join(['%.2f,%.2f'%(a,b) for a,b in default_res])))
        params = copy.deepcopy(default_params)
        res_list = [('%s=%s'%(param_name,choice[0]), default_res)]
        for c in choice[1:]:
            params[param_name] = (param_type, c)
            res = loop_rosbag(params)
            res_list.append(('%s=%s'%(param_name,c), res))
            fp.write('|%s|%s|\n'%(c,'|'.join(['%.2f,%.2f'%(a,b) for a,b in res])))
        fp.write('\n')

        global png_idx
        png_name = os.path.join(png_dir, '%06d.jpg'%png_idx)
        plot_rmse(res_list, png_name)
        file_name = '![fig](' + root_dir + "/cache/figure/%06d.jpg)\n"
        fp.write(file_name%(png_idx))
        fp.flush()
        png_idx = png_idx + 1

if __name__ == '__main__':

    # the first choice always be default
    # param_table = [('use_larvio_flag', 'int', ['0','1']),
    #                ('max_features_in_one_grid', 'int', ['0','1'])]
    # param_table = [('use_larvio_flag', 'int', ['0','1'])]
    param_table = [('use_left_perturbation_flag', 'int', ['0','1'])]

    default_params = {a:(b,c[0]) for a,b,c in param_table}

    # clean roslog
    os.system('rm -rf ~/.ros/log')

    # record result in markdown
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)
        
    fp = open(res_table,'w')
    fp.write('# OrcVIO evaluation on EuROC dataset\n')
    fp.write('NOTE: we log the RMSE of orientation and postion. Unit: [deg, m]\n')

    # single variable-controlling
    fp.write('## Single param comparision\n')
    params = copy.deepcopy(default_params)
    batch_run_single_change(param_table, params, fp)

