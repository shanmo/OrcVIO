import numpy as np

import path_def 

seq_names = ['0001', '0019', '0022', '0023', '0035', '0036', '0039', '0061', '0064', '0093']

kitti_date = '2011_09_26'

total_gt_num = 0
total_pred_num = 0

total_30_05_num = 0
total_30_10_num = 0
total_30_15_num = 0

total_45_05_num = 0
total_45_10_num = 0
total_45_15_num = 0

total_inf_05_num = 0
total_inf_10_num = 0
total_inf_15_num = 0

for kitti_drive in seq_names:

    kitti_dir = kitti_date + '_' + kitti_drive
    pr_table_dir = path_def.kitti_cache_path + kitti_dir + '/evaluation/'

    pr_filename = pr_table_dir + 'object_eval.txt'

    with open(pr_filename) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    # print(content)

    # TODO: FIXME: these indices eg 21, 22 may not be correct 
    gt_num = content[21].split(' ')[-1]
    # print(gt_num)
    total_gt_num += float(gt_num)

    pred_num = content[22].split(' ')[-1]
    # print(pred_num)
    total_pred_num += float(pred_num)

    # < 30

    num_30_05 = content[24].split(' ')[-1]
    # print(num_30_05)
    total_30_05_num += float(num_30_05)

    num_30_10 = content[25].split(' ')[-1]
    total_30_10_num += float(num_30_10)

    num_30_15 = content[26].split(' ')[-1]
    total_30_15_num += float(num_30_15)

    # < 45

    num_45_05 = content[28].split(' ')[-1]
    total_45_05_num += float(num_45_05)

    num_45_10 = content[29].split(' ')[-1]
    total_45_10_num += float(num_45_10)

    num_45_15 = content[30].split(' ')[-1]
    total_45_15_num += float(num_45_15)

    # < inf

    num_inf_05 = content[32].split(' ')[-1]
    total_inf_05_num += float(num_inf_05)

    num_inf_10 = content[33].split(' ')[-1]
    total_inf_10_num += float(num_inf_10)

    num_inf_15 = content[34].split(' ')[-1]
    total_inf_15_num += float(num_inf_15)

print("30 deg 0.5 m precision {}".format(total_30_05_num / total_pred_num))
print("30 deg 0.5 m recall {}".format(total_30_05_num / total_gt_num))

print("30 deg 1.0 m precision {}".format(total_30_10_num / total_pred_num))
print("30 deg 1.0 m recall {}".format(total_30_10_num / total_gt_num))

print("30 deg 1.5 m precision {}".format(total_30_15_num / total_pred_num))
print("30 deg 1.5 m recall {}".format(total_30_15_num / total_gt_num))

print("\n")
print("\n")
print("\n")


print("45 deg 0.5 m precision {}".format(total_45_05_num / total_pred_num))
print("45 deg 0.5 m recall {}".format(total_45_05_num / total_gt_num))

print("45 deg 1.0 m precision {}".format(total_45_10_num / total_pred_num))
print("45 deg 1.0 m recall {}".format(total_45_10_num / total_gt_num))

print("45 deg 1.5 m precision {}".format(total_45_15_num / total_pred_num))
print("45 deg 1.5 m recall {}".format(total_45_15_num / total_gt_num))

print("\n")
print("\n")
print("\n")

print("inf deg 0.5 m precision {}".format(total_inf_05_num / total_pred_num))
print("inf deg 0.5 m recall {}".format(total_inf_05_num / total_gt_num))

print("inf deg 1.0 m precision {}".format(total_inf_10_num / total_pred_num))
print("inf deg 1.0 m recall {}".format(total_inf_10_num / total_gt_num))

print("inf deg 1.5 m precision {}".format(total_inf_15_num / total_pred_num))
print("inf deg 1.5 m recall {}".format(total_inf_15_num / total_gt_num))
