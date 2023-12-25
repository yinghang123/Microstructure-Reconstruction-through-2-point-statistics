import numpy as np
import cv2
from PreprocessingFunction_2 import db_cluster, dis3d

"Establishing the linkage of color---euler angle, for visualization generating input file for Crystal plasticity model"

ipfcolor = np.loadtxt(open(r"C:\Users\liuyh\Desktop\ebsd\1\All_color-BC + IPF + GB.csv", "rb"), delimiter=",",
                      skiprows=2, usecols=[4, 5, 6])
euler_real = np.loadtxt(open(r"C:\Users\liuyh\Desktop\ebsd\1\Allcolor_euler.csv", "rb"), delimiter=",", skiprows=1,
                        usecols=[0,1,2])

ipfcolor_list = []
for i in range(ipfcolor.shape[0]):
    ipfcolor_list.append(list(ipfcolor[i]))

img = cv2.imread('original_voronoi_ild.tif') # this is the input microstructure, the color of it is based on euler angle (needing further regulation)
labels, cluster_number, euler_cluster_center = db_cluster(img)

########### convert euler angle to ipfcolor ############

grain_ave_ipf = []
print('convert euler angle to ipfcolor')
for i in range(len(euler_cluster_center)):
    euler_distancelist = []
    for j in range(euler_real.shape[0]):
        cur_euler_dis = dis3d(euler_cluster_center[i], euler_real[j])
        euler_distancelist.append(cur_euler_dis)
    min_index = euler_distancelist.index(min(euler_distancelist))
    grain_ave_ipf.append([ipfcolor[min_index][2], ipfcolor[min_index][1], ipfcolor[min_index][0]])

center = grain_ave_ipf
print('ipf center:', center)
np.save('centercolor.npy', center) ### 是欧拉中心所对应的ipf中心颜色 average ipf color of a grain
print('successful generation of color file')

