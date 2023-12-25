import time
import cv2
from PreprocessingFunction_2 import db_cluster, generate_eignmicro
from HelperFunctions_StochasticGeneration import *
from StochasticGeneration2 import *

start = time.perf_counter()
img = cv2.imread('original_voronoi_ild.tif') # this is the input microstructure, the color of it is based on euler angle (needing further regulation)
labels, cluster_number, euler_cluster_center = db_cluster(img)
print('dbscan cluster number (i.e. numbers of variants considered)', cluster_number)
print('labels shape (picture shape)', labels.shape)
center = np.load('centercolor.npy')

# to show the optimized microstructure
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        cur_color = center[labels[i][j]]
        img[i][j][0] = cur_color[0]
        img[i][j][1] = cur_color[1]
        img[i][j][2] = cur_color[2]
cv2.imshow('img_origin_voronoi_clustered', img) # this image has been clustered for further generation
cv2.waitKey(0)
cv2.imwrite('original_voronoi_cluster.tif', img)

# sample the fewest phase ## maybe no need
k = [0 for i in range(cluster_number)]
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        number = labels[i][j]
        k[number] += 1

# print('number of each cluster pixels',k)
# original_color_fraction = np.c_[center, k]
# np.save('original_color_fraction.npy', original_color_fraction)

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
                labels[i][j] = min_k_index  # change the fewest phase to the first place

start = time.perf_counter()
eignmicro = generate_eignmicro(labels)
struct = np.array(eignmicro, dtype=np.float64)
generator = EigenGenerator_BaseClass  # generate microstructure through 2pt stats
stats = twopointstats(struct)
sum1 = stats.sum()
print('sum1',sum1)

# generate the microstructure
gen = generator(stats, 'complete')
gen.filter('flood', alpha=0.3, beta=0.35)
sampled_micro_1, sampled_micro_2 = gen.generate()
end = time.perf_counter()

#### change the position back ###
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

cv2.imshow('generated_microstructure', img) ######################## You can import it into aztech to analyze the center and growth vectors and then use anisotropic voronoi as post processing.
cv2.waitKey(0)
cv2.imwrite('generated_microstructure.tif', img) # this is the generated microstructure but need post processing


