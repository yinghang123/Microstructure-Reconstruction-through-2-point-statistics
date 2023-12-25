import math
from sklearn.cluster import KMeans, DBSCAN
import cv2
from HelperFunctions_StochasticGeneration import *

"""using the preprocessing to clean up, and generate from 2-pt stat"""


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
    print('dbscan clustering over, the microstructure now is in label form')
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

