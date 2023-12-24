import math
from sklearn.cluster import KMeans, DBSCAN
from StochasticGeneration2 import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from HelperFunctions_StochasticGeneration import *
import time

"""using the """


def dis3d(a, b):
    dis = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    dis = math.sqrt(dis)
    return dis


def clustercolor(path, cluster_number):
    """KMeans method is used to cluster the color of one variant with different colors"""

    img = cv2.imread(path)
    print('Ori-dimension', img.shape)
    img0 = img.copy()
    width = img.shape[1]
    height = img.shape[0]
    img = np.reshape(img, (width * height, 3))

    cluster = KMeans(n_clusters=cluster_number, random_state=0)
    cluster = cluster.fit(img)
    center = cluster.cluster_centers_
    center = np.array(center, dtype=np.uint8)
    print(center)
    np.save('centercolor.npy', center)
    center = list(center)
    for i in range(len(center)):
        center[i] = list(center[i])
        center[i] = tuple([int(x) for x in center[i]])
    labels = cluster.predict(img)

    img = img.tolist()
    for i in range(len(img)):
        img[i].append(labels[i]+1)
    img = np.array(img)
    np.savetxt(".\\generated_dataset\\pixel_euler_cluster.csv", img, delimiter=" ", fmt='%d')

    clusters = {}
    n = 0
    for item in labels:
        if item in clusters:
            clusters[item].append(img[n])
        else:
            clusters[item] = [img[n]]
        n += 1
    labels = np.reshape(labels, (height, width))
    return labels, center


def db_cluster(img):
    """using dbscan method to cluster the color of one variant with different grain colors"""

    print('Ori-dimension', img.shape)
    img0 = img.copy()
    width = img.shape[1]
    height = img.shape[0]
    img = np.reshape(img, (width * height, 3))
    img2 = img.copy()
    db = DBSCAN(eps=4, min_samples=1).fit(img)
    labels = db.labels_
    img = img.tolist()
    ##########   visualization of cluster process   ###############
    for i in range(len(img)):
        img[i].append(labels[i] + 1)
    img = np.array(img)
    np.savetxt(".\\generated_dataset\\pixel_euler_cluster.csv", img, delimiter=" ", fmt='%d')
    centernumber = max(labels) + 1

    ###########     find the center color using KMeans     ###########
    each_cluster_euler = [[] for i in range(centernumber)]
    for i in range(len(labels)):
        cur_cluster_belong = labels[i]
        imgi = img2[i].tolist()
        if imgi not in each_cluster_euler[cur_cluster_belong]:
            each_cluster_euler[cur_cluster_belong].append(imgi)

    center_of_euler = []
    for i in range(len(each_cluster_euler)):
        cluster = KMeans(n_clusters=1, random_state=0)
        cluster = cluster.fit(each_cluster_euler[i])
        center = cluster.cluster_centers_
        center = np.array(center, dtype=np.uint8)
        center_of_euler.append(center[0])
    print('center_of_euler', center_of_euler)
    '''k = [0 for i in range(centernumber)]
    for i in range(len(labels)):
        k[labels[i]] += 1
    print(k)'''
    print('dbscan clustering over')
    labels = np.reshape(labels, (height, width))
    return labels, centernumber, center_of_euler


###################### 有了label就能化为eigen了: to the eigenmicrostructure for GRF input ##########################
def generate_eignmicro(labels):
    shape = labels.shape
    labelsline = np.reshape(labels, (labels.shape[0] * labels.shape[1]))
    shape = list(shape)
    third_dem = max(labelsline)+1
    shape.append(third_dem)
    eignmicro = np.zeros(shape, dtype=np.uint8)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            index_of_label = labels[i][j]
            eignmicro[i][j][index_of_label] = 1
    return eignmicro


#def generate_from(euler_img):


################## generate #######################################
ipfcolor = np.loadtxt(open(r"C:\Users\liuyh\Desktop\ebsd\1\All_color-BC + IPF + GB.csv", "rb"), delimiter=",",
                      skiprows=2, usecols=[4, 5, 6])
euler_real = np.loadtxt(open(r"C:\Users\liuyh\Desktop\ebsd\1\Allcolor_euler.csv", "rb"), delimiter=",", skiprows=1,
                        usecols=[0,1,2])

"""Establishing the linkage of color---euler angle, for generating input file for Crystal plasticity model"""

ipfcolor_list = []
for i in range(ipfcolor.shape[0]):
    ipfcolor_list.append(list(ipfcolor[i]))

start = time.perf_counter()
cluster_number = 12 # 12 kinds of variants in Ti martensite
#labels, center = clustercolor('original_voronoi_ild.tif', cluster_number)
img = cv2.imread('original_voronoi_ild.tif')


labels, cluster_number, euler_cluster_center = db_cluster(img)
print('dbscan cluster number', cluster_number)
print('labels shape',labels.shape)

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
print(center)
np.save('centercolor.npy', center) ### 是欧拉中心所对应的ipf中心颜色
#center = euler_cluster_center
### Clustered the original image
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        cur_color = center[labels[i][j]]
        img[i][j][0] = cur_color[0]
        img[i][j][1] = cur_color[1]
        img[i][j][2] = cur_color[2]
cv2.imshow('img_origin_voronoi', img)
cv2.waitKey(0)
cv2.imwrite('original_voronoi_cluster.tif', img)

### 对相分数最小的采样，需要调换labels## 现在看来没必要
#print('labels', labels)
k = [0 for i in range(cluster_number)]
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        number = labels[i][j]
        k[number] += 1
print(k)
original_color_fraction = np.c_[center, k]
np.save('original_color_fraction.npy', original_color_fraction)


change_or_not = True
if min(k) == k[0]:
    change_or_not = False

if change_or_not:
    min_k_index = k.index(min(k))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i][j] == min_k_index:
                labels[i][j] = 0
            elif labels[i][j] == 0:
                labels[i][j] = min_k_index


start = time.perf_counter()
eignmicro = generate_eignmicro(labels)
struct = np.array(eignmicro, dtype=np.float64)
generator = EigenGenerator_BaseClass
stats = twopointstats(struct)
sum1 = stats.sum()
print('sum1',sum1)

''' plot gaussian field
auto = autocorrelations(struct)
auto = auto[:,:,0:1]
xx = int(0.5 * auto.shape[0])
yy = int(0.5 * auto.shape[1])
plotauto = auto.copy()
# 左上移到右下
for i in range(xx, auto.shape[0]):
    for j in range(yy, auto.shape[1]):
        plotauto[i][j] = auto[i-xx][j-yy]
for i in range(0, xx):
    for j in range(0, yy):
        plotauto[i][j] = auto[i+xx][j+yy]
for i in range(xx, auto.shape[0]):
    for j in range(0, yy):
        plotauto[i][j] = auto[i-xx][j+yy]
for i in range(0, xx):
    for j in range(yy, auto.shape[1]):
        plotauto[i][j] = auto[i+xx][j-yy]
print('autoshape', auto.shape)
#f = plt.figure(figsize=[8, 4.5])
f = plt.figure()
ax1 = plt.axes(projection='3d')
x = np.arange(-yy,yy,1)
y = np.arange(-xx,xx,1)
X, Y = np.meshgrid(x, y)
shape = plotauto.shape[0:2]
plotauto2 = np.zeros(shape)
for i in range(plotauto2.shape[0]):
    for j in range(plotauto2.shape[1]):
        plotauto2[i][j] = plotauto[i][j][0]
print(X.shape, Y.shape, plotauto2.shape)
ax1.plot_surface(X,Y,plotauto2, rstride=1, cstride=1, cmap='rainbow')
ax1.contourf(X,Y,plotauto2,zdir='z', offset=6,cmap="rainbow")
#ax1.set_zlim(-0.03, 0.07)
#ax1.imshow(plotauto)
#plt.xlim(-112, 112)
#plt.ylim(-58, 58)
#plt.axis('off')
plt.savefig('gaussian.png')
plt.show()

f = plt.figure()
ax2 = f.add_subplot(121)
ax2.imshow(plotauto)
plt.axis('off')
plt.savefig('2d.png')
plt.show()'''

# generate the microstructure
gen = generator(stats, 'complete')
gen.filter('flood', alpha=0.3, beta=0.35)
sampled_micro_1, sampled_micro_2 = gen.generate()
end = time.perf_counter()

#### 将sample调换回来 ###
if change_or_not:
    for i in range(sampled_micro_1.shape[0]):
        for j in range(sampled_micro_1.shape[1]):
            if sampled_micro_1[i][j][min_k_index] == 1:
                sampled_micro_1[i][j][min_k_index] = 0
                sampled_micro_1[i][j][0] = 1
            elif sampled_micro_1[i][j][0] == 1:
                sampled_micro_1[i][j][0] = 0
                sampled_micro_1[i][j][min_k_index] = 1


stats = twopointstats(sampled_micro_1)
sum2 = stats.sum()
print('sum2', sum2)
print('rmse', (sum1-sum2)/sum1)

shape = list(labels.shape)
shape.append(3)
img = np.zeros(shape,dtype=np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for p in range(sampled_micro_1.shape[2]):
            if sampled_micro_1[i][j][p] == 1:
                img[i][j][0] = center[p][0]
                img[i][j][1] = center[p][1]
                img[i][j][2] = center[p][2]

print('generating time', end-start, 'seconds')

cv2.imshow('img', img) ######################## You can import it into aztech to analyze the center and growth vectors and then use anisotropic voronoi to de-isolate it.
cv2.waitKey(0)
cv2.imwrite('kali_gen.tif', img)

