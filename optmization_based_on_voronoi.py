import math
import skimage
from export_grow_vectors_of_each_grain import *
from sklearn.cluster import DBSCAN


def island_points(points_set, center):
    """identifies the isolated part of one lath caused by excessive growth"""

    imgdst = np.zeros(list(img0.shape)[0:2], dtype=np.uint8)
    for i in range(len(points_set)):
        x = points_set[i][0]
        y = points_set[i][1]
        for p in range(imgdst.shape[0]):
            for q in range(imgdst.shape[1]):
                if p == x and q == y:
                    imgdst[p][q] = 255
    label_map = skimage.measure.label(imgdst)
    regions = skimage.measure.regionprops(label_map)
    island_points_list = []
    for i in range(len(regions)):
        regpts = regions[i].coords
        regpts = list(regpts)
        regpts = [tuple(x) for x in regpts]
        if center not in regpts:
            island_points_list.extend(regpts)
    return island_points_list


def dis3d(a, b):
    dis = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    dis = math.sqrt(dis)
    return dis


def ptsonline(p1, p2):
    """"using a line to identify the isolated points"""
    pts = []
    if p1[0] == p2[0]:
        for i in range(min(p1[1], p2[1]), max(p1[1], p2[1])):
            pts.append((p1[0], i))
            # if (p1[0]-1) in range(0, img0.shape[0]):
            # pts.append((p1[0]-1, i))
            # if (p1[0] + 1) in range(0, img0.shape[0]):
            # pts.append((p1[0] + 1, i))

    elif p1[1] == p2[1]:
        for i in range(min(p1[0], p2[0]), max(p1[0], p2[0])):
            pts.append((i, p1[1]))
            # if (p1[1]-1) in range(0, img0.shape[1]):
            # pts.append((i, p1[1]-1))
            # if (p1[1] + 1) in range(0, img0.shape[1]):
            # pts.append((i, p1[1] + 1))

    else:
        k = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p1[1] - k * p1[0]
        if abs(p1[0] - p2[0]) >= abs(p1[1] - p2[1]):
            for i in range(min(p1[0], p2[0]), max(p1[0], p2[0])):
                y_real = k * i + b
                y = int(y_real + 0.5)
                pts.append((i, y))
                # if (y+1) in range(0, img0.shape[1]):
                # pts.append((i, y+1))
                # if (y - 1) in range(0, img0.shape[1]):
                # pts.append((i, y-1))
        else:
            for i in range(min(p1[1], p2[1]), max(p1[1], p2[1])):
                x_real = (i - b) / k
                x = int(x_real + 0.5)
                pts.append((x, i))
                # if (x + 1) in range(0, img0.shape[0]):
                # pts.append((x+1, i))
                # if (x - 1) in range(0, img0.shape[0]):
                # pts.append((x-1, i))
    return pts


# define some necessary variables
def distance(p1, p2):
    length = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return length


def vector_value(pp):
    value = math.sqrt(pp[0] ** 2 + pp[1] ** 2)
    if value == 0:
        value += 0.1
    return value


def uniform_vector(pp):
    q = np.array(pp, dtype=np.float64).copy()
    value = vector_value(pp)
    q[0] = pp[0] / value
    q[1] = pp[1] / value
    return np.array(q, dtype=np.float64)


def grow_vector(direction, r):
    x = direction[0] * r / vector_value(direction)
    y = direction[1] * r / vector_value(direction)
    return np.array((x, y), dtype=np.float64)


def velocity(current, center):
    global dict_center
    current_array = np.array(current, dtype=np.float64)
    center_array = np.array(center, dtype=np.float64)
    delta = current_array - center_array
    er = uniform_vector(delta)
    direction_of_grow = dict_center[center][0]
    r = dict_center[center][1]
    pi = grow_vector(direction_of_grow, r)
    ep = uniform_vector(pi)
    left = math.sqrt(1 / (math.pi * r))
    right = (1 + (1 / (r ** 2) - 1) * (np.dot(ep, er) ** 2)) ** (-1 / 2)
    return right * left


def voronoi_optimization_softwareDATA(data):
    """To some extent, one can directly import the five elements from official software analysing EBSD data,
    the form of the input should be consistent with the 3 files above,
    here, preprocessing is carried out using this function"""

    global dict_center
    img = cv2.imread('kali_gen_sep.tif')
    imageshape = list(img.shape)[0:2]
    r = []
    for i in range(data.shape[0]):
        r.append(data[i][5])
    center = []
    for i in range(data.shape[0]):
        ctrx = int(data[i][1] / 0.4 + 0.5)
        ctry = int(data[i][0] / 0.4 + 0.5)
        center.append((ctrx, ctry))
    direction = []
    for i in range(data.shape[0]):
        theta = data[i][4]
        rad = math.pi * theta / 180
        tantheta = np.tan(rad)
        cur_direction = np.array([-10 * tantheta, 10], dtype=np.float64)
        direction.append(cur_direction)
    euler_ave = []
    for i in range(data.shape[0]):
        euler1, euler2, euler3 = data[i][6], data[i][7], data[i][8]
        euler_ave.append((euler1, euler2, euler3))
    grain_ave_ipf = []  # from euler angles to ipf colors
    print('finding euler-ipf relation')
    for i in range(len(euler_ave)):
        euler_distancelist = []
        for j in range(euler_real.shape[0]):
            cur_euler_dis = dis3d(euler_ave[i], euler_real[j])
            euler_distancelist.append(cur_euler_dis)
        min_index = euler_distancelist.index(min(euler_distancelist))
        grain_ave_ipf.append([ipfcolor[min_index][2], ipfcolor[min_index][1], ipfcolor[min_index][0]])

    center_color = np.load('centercolor.npy')
    img_fraction = cv2.imread('kali_gen.tif')
    fake_fraction = [0 for i in range(center_color.shape[0])]
    for i in range(img_fraction.shape[0]):
        for j in range(img_fraction.shape[1]):
            for p in range(center_color.shape[0]):
                if img_fraction[i][j][0] == center_color[p][0] and img_fraction[i][j][1] == center_color[p][1] and \
                        img_fraction[i][j][2] == center_color[p][2]:
                    number = p
            fake_fraction[number] += 1
    print('phase fraction of generated sample', fake_fraction)
    color_fraction = np.load('original_color_fraction.npy')
    area_list = []  # adjust the area to fit the original fraction
    for i in range(data.shape[0]):
        w = data[i][2] / 0.4
        h = data[i][3] / 0.4
        s = math.pi * w * h * 0.25
        # s = data[i][9]
        for j in range(color_fraction.shape[0]):
            if grain_ave_ipf[i][0] == color_fraction[j][0] and grain_ave_ipf[i][1] == color_fraction[j][1] and \
                    grain_ave_ipf[i][2] == color_fraction[j][2]:
                magnification = color_fraction[j][3] / fake_fraction[j]
                s = s * magnification
        area_list.append(s)

    ### realize
    centerpts = center
    dict_center = {}
    for i in range(len(centerpts)):
        dict_center[centerpts[i]] = [direction[i], r[i]]

    colorlist = []

    for i in range(len(centerpts)):
        colorlist.append(grain_ave_ipf[i])

    dict_color = {}
    for i in range(len(centerpts)):
        dict_color[centerpts[i]] = np.array(colorlist[i], dtype=np.uint8)

    imageshape.append(3)
    img0 = np.zeros(imageshape, dtype=np.uint8)
    img0.fill(255)

    timeset_center = {}
    for p in range(len(centerpts)):
        timeset_center[centerpts[p]] = []

    eachij_center_tm = {}
    tmlist = []
    for i in range(img0.shape[0]):
        print('Browsing line:', i)
        for j in range(img0.shape[1]):
            dict_time = {}
            for p in range(len(centerpts)):
                leng = distance((i, j), centerpts[p])
                if leng == 0:
                    tm = 0
                else:
                    vel = velocity((i, j), centerpts[p]) * math.sqrt(area_list[p])
                    tm = leng / vel
                dict_time[tm] = centerpts[p]
            eachij_center_tm[(i, j)] = dict_time
            min_time = min(dict_time.keys())
            tmlist.append(min_time)
            belong_center = dict_time[min_time]
            timeset_center[belong_center].append([min_time, (i, j)])
            ptscolor = dict_color[belong_center]
            img0[i][j] = ptscolor
            ############# threshold the time to show the growing process ########################
            '''if min_time > 1.0:
                img0[i][j] = [255,255,255]
            else:
                pass'''
    print('max_time', max(tmlist))

    print('Browsing over, finding rejected points')
    # cv2.imshow('before-line', img0)

    dst = img0.copy()
    dst.fill(255)
    reject_points = []
    for i in range(len(centerpts)):
        print('Browsing center:', i)
        cur_center_ijbelong = []
        for j in range(len(timeset_center[centerpts[i]])):
            cur_center_ijbelong.append(timeset_center[centerpts[i]][j][1])
        islpts = island_points(cur_center_ijbelong, centerpts[i])
        reject_points.extend(islpts)

    # print('size of rejdom:', len(reject_points))
    rejdom = img0.copy()
    rejdom.fill(255)

    for rej in reject_points:
        rejdom[rej[0]][rej[1]][0] = 0
        rejdom[rej[0]][rej[1]][1] = 0
        rejdom[rej[0]][rej[1]][2] = 0
        rej_tms_and_centers = eachij_center_tm[rej]
        min_time = min(rej_tms_and_centers.keys())
        del rej_tms_and_centers[min_time]
        min_time = min(rej_tms_and_centers.keys())
        belong_center = rej_tms_and_centers[min_time]
        ptscolor = dict_color[belong_center]
        img0[rej[0]][rej[1]] = ptscolor

    cv2.imshow('after-ild', img0)
    # cv2.imshow('rej-ild', rejdom)
    cv2.waitKey(0)
    cv2.imwrite('final_result.tif', img0)


def voronoi_optimization(img):
    """post-processing using aniso-voronoi optimization to smooth the boundaries and fit the grain morphology
    when using as pre-processing, the img should be clustered first"""

    global dict_center
    imageshape = list(img.shape)[0:2]
    num_img = img_to_num(img)
    aspect_ratio, center_points, tilt_ang, wh, graincolor = five_elements(num_img)
    ###  origin_r
    r = []
    for i in range(len(aspect_ratio)):
        r.append(aspect_ratio[i])

    ###  center
    center = []
    for i in range(len(center_points)):
        ctrx = int(center_points[i][1] + 0.5)
        ctry = int(center_points[i][0] + 0.5)
        center.append((ctrx, ctry))

    ### direction
    direction = []
    for i in range(len(tilt_ang)):
        theta = tilt_ang[i]
        rad = math.pi * theta / 180
        tantheta = np.tan(rad)
        cur_direction = np.array([-10 * tantheta, 10], dtype=np.float64)
        direction.append(cur_direction)

    ### calibrate the volume fractions of each local state
    color_fraction = np.load('original_color_fraction.npy')  # this is the original volume fraction
    # now calculate the f of samples
    center_color = np.load('centercolor.npy')
    img_fraction = cv2.imread('kali_gen_sep.tif')
    fake_fraction = [0 for i in range(center_color.shape[0])]
    for i in range(img_fraction.shape[0]):
        for j in range(img_fraction.shape[1]):
            for p in range(center_color.shape[0]):
                if img_fraction[i][j][0] == center_color[p][0] and img_fraction[i][j][1] == center_color[p][1] and \
                        img_fraction[i][j][2] == center_color[p][2]:
                    number = p
            fake_fraction[number] += 1
    print('fake_fraction', fake_fraction)
    magnification = [color_fraction[i][3] / fake_fraction[i] for i in range(center_color.shape[0])]
    area_list = []
    for i in range(len(wh)):
        w = wh[i][0]
        h = wh[i][1]
        s = math.pi * w * h * 0.25
        ### rearrange the area
        color_of_i = graincolor[i]
        for j in range(center_color.shape[0]):
            if color_of_i[0] == center_color[j][0] and color_of_i[1] == center_color[j][1] and color_of_i[2] == \
                    center_color[j][2]:
                s = s * magnification[j]
        area_list.append(s)
    ###################  realize  ###########################
    centerpts = center
    dict_center = {}
    for i in range(len(centerpts)):
        dict_center[centerpts[i]] = [direction[i], r[i]]

    colorlist = []
    for i in range(len(centerpts)):
        colorlist.append(graincolor[i])

    dict_color = {}
    for i in range(len(centerpts)):
        dict_color[centerpts[i]] = np.array(colorlist[i], dtype=np.uint8)

    imageshape.append(3)
    img0 = np.zeros(imageshape, dtype=np.uint8)
    img0.fill(255)

    timeset_center = {}
    for p in range(len(centerpts)):
        timeset_center[centerpts[p]] = []

    eachij_center_tm = {}
    tmlist = []
    for i in range(img0.shape[0]):
        print('Browsing line', i)
        for j in range(img0.shape[1]):
            dict_time = {}
            for p in range(len(centerpts)):
                leng = distance((i, j), centerpts[p])
                if leng == 0:
                    tm = 0
                else:
                    vel = velocity((i, j), centerpts[p]) * math.sqrt(area_list[p])
                    tm = leng / vel
                dict_time[tm] = centerpts[p]
            eachij_center_tm[(i, j)] = dict_time
            min_time = min(dict_time.keys())
            tmlist.append(min_time)
            belong_center = dict_time[min_time]
            timeset_center[belong_center].append([min_time, (i, j)])
            ptscolor = dict_color[belong_center]
            img0[i][j] = ptscolor

            '''if min_time > 1.0:
                img0[i][j] = [255,255,255]
            else:
                pass'''
    # print('max_time', max(tmlist))

    print('Browsing over, finding rejected_points')
    # cv2.imshow('before-line', img0)

    dst = img0.copy()
    dst.fill(255)
    reject_points = []
    for i in range(len(centerpts)):
        print('Finding island', i)
        cur_center_ijbelong = []
        for j in range(len(timeset_center[centerpts[i]])):
            cur_center_ijbelong.append(timeset_center[centerpts[i]][j][1])
        islpts = island_points(cur_center_ijbelong, centerpts[i])
        reject_points.extend(islpts)

    print('Size of rej', len(reject_points))
    rejdom = img0.copy()
    rejdom.fill(255)

    for rej in reject_points:
        rejdom[rej[0]][rej[1]][0] = 0
        rejdom[rej[0]][rej[1]][1] = 0
        rejdom[rej[0]][rej[1]][2] = 0
        rej_tms_and_centers = eachij_center_tm[rej]
        min_time = min(rej_tms_and_centers.keys())
        del rej_tms_and_centers[min_time]
        min_time = min(rej_tms_and_centers.keys())
        belong_center = rej_tms_and_centers[min_time]
        ptscolor = dict_color[belong_center]
        img0[rej[0]][rej[1]] = ptscolor

    # img0 = cv2.resize(img0, (int(3 * img0.shape[1]), int(3 * img0.shape[0])), interpolation=cv2.INTER_NEAREST)
    # rejdom = cv2.resize(rejdom, (int(3 * rejdom.shape[1]), int(3 * rejdom.shape[0])), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('after-ild', img0)
    # cv2.imshow('rej-ild', rejdom)
    cv2.waitKey(0)
    # cv2.imwrite('final_result.tif', img0)
    return img0


def cluster_img_dbscan(img_arr):
    """make the colors within a single grain uniform; not so suitable for pre-processing
    pre-processing should cluster the euler angles"""

    width, height, channels = img_arr.shape
    img_arr_2d = img_arr.reshape(width * height, channels)

    dbscan = DBSCAN(eps=4, min_samples=1).fit(img_arr_2d)

    cluster_labels = np.asarray(dbscan.labels_, dtype=np.uint8)
    img_arr_compressed = np.zeros_like(img_arr_2d)
    for label in np.unique(cluster_labels):
        if label == -1:
            img_arr_compressed[cluster_labels == label] = [0, 0, 0]
        else:
            img_arr_compressed[cluster_labels == label] = np.mean(img_arr_2d[cluster_labels == label], axis=0)
            print(np.mean(img_arr_2d[cluster_labels == label], axis=0))
    img_arr_compressed = img_arr_compressed.reshape(width, height, channels)
    cv2.imwrite('cluster_img.tif', img_arr_compressed)
    return img_arr_compressed


def preprocessing_voronoi_euler(data):
    """fill the array with euler angles suitable for latter operation
    here, the euler angles are clustered instead of ipf color,
    this process is similar to the first one, the microstructure used in generate file is obtained here"""

    global dict_center, imageshape
    r = []
    for i in range(data.shape[0]):
        r.append(data[i][5])

    center = []
    for i in range(data.shape[0]):
        ctrx = int(data[i][1] / 0.4 + 0.5)
        ctry = int(data[i][0] / 0.4 + 0.5)
        center.append((ctrx, ctry))

    area_list = []
    for i in range(data.shape[0]):
        w = data[i][2] / 0.4
        h = data[i][3] / 0.4
        s = math.pi * w * h * 0.25
        # s = data[i][9]
        area_list.append(s)

    direction = []
    for i in range(data.shape[0]):
        theta = data[i][4]
        rad = math.pi * theta / 180
        tantheta = np.tan(rad)
        cur_direction = np.array([-10 * tantheta, 10], dtype=np.float64)
        direction.append(cur_direction)

    euler_ave = []  # filling the img_shape array with average euler angle
    for i in range(data.shape[0]):
        euler1, euler2, euler3 = data[i][6], data[i][7], data[i][8]
        euler_ave.append((euler1, euler2, euler3))

    ###################  realize  ###########################
    centerpts = center
    dict_center = {}
    for i in range(len(centerpts)):
        dict_center[centerpts[i]] = [direction[i], r[i]]

    colorlist = []
    for i in range(len(centerpts)):
        colorlist.append(euler_ave[i])

    dict_color = {}
    for i in range(len(centerpts)):
        dict_color[centerpts[i]] = np.array(colorlist[i], dtype=np.uint8)

    imageshape.append(3)
    img0 = np.zeros(imageshape, dtype=np.uint8)
    img0.fill(255)
    # img0 = img0[0:400, 0:400]

    timeset_center = {}
    for p in range(len(centerpts)):
        timeset_center[centerpts[p]] = []

    eachij_center_tm = {}
    tmlist = []
    for i in range(img0.shape[0]):
        print('Browsing line', i, 'total line:', imageshape[0] - 1)
        for j in range(img0.shape[1]):
            dict_time = {}
            for p in range(len(centerpts)):
                leng = distance((i, j), centerpts[p])
                if leng == 0:
                    tm = 0
                else:
                    vel = velocity((i, j), centerpts[p]) * math.sqrt(area_list[p])
                    tm = leng / vel
                dict_time[tm] = centerpts[p]
            eachij_center_tm[(i, j)] = dict_time
            min_time = min(dict_time.keys())
            tmlist.append(min_time)
            belong_center = dict_time[min_time]
            timeset_center[belong_center].append([min_time, (i, j)])
            ptscolor = dict_color[belong_center]
            img0[i][j] = ptscolor

            '''if min_time > 1.0:
                img0[i][j] = [255,255,255]
            else:
                pass'''
    print('max_time', max(tmlist))

    print('Browsing over, finding rejected_points')
    # cv2.imshow('before-line', img0)1

    dst = img0.copy()
    dst.fill(255)
    reject_points = []
    for i in range(len(centerpts)):
        print('Finding island No.', i)
        cur_center_ijbelong = []
        for j in range(len(timeset_center[centerpts[i]])):
            cur_center_ijbelong.append(timeset_center[centerpts[i]][j][1])
        islpts = island_points(cur_center_ijbelong, centerpts[i])
        reject_points.extend(islpts)

    print('Size of rej', len(reject_points))
    rejdom = img0.copy()
    rejdom.fill(255)

    for rej in reject_points:
        rejdom[rej[0]][rej[1]][0] = 0
        rejdom[rej[0]][rej[1]][1] = 0
        rejdom[rej[0]][rej[1]][2] = 0
        rej_tms_and_centers = eachij_center_tm[rej]
        min_time = min(rej_tms_and_centers.keys())
        del rej_tms_and_centers[min_time]
        min_time = min(rej_tms_and_centers.keys())
        belong_center = rej_tms_and_centers[min_time]
        ptscolor = dict_color[belong_center]
        img0[rej[0]][rej[1]] = ptscolor

    # img0 = cv2.resize(img0, (int(3 * img0.shape[1]), int(3 * img0.shape[0])), interpolation=cv2.INTER_NEAREST)
    # rejdom = cv2.resize(rejdom, (int(3 * rejdom.shape[1]), int(3 * rejdom.shape[0])), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('optimized microstructure', img0)
    # cv2.imshow('rej-ild', rejdom)
    cv2.waitKey(0)
    cv2.imwrite('original_voronoi_ild.tif', img0)
    return img0


data = np.loadtxt(open("Subset1_sythetic-Grain List.csv", "rb"), delimiter=",", skiprows=2,
                  usecols=[4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 2])
# the relationship between ipf colors and euler angles should be established
ipfcolor = np.loadtxt(open(r"Subset1_sythetic-IPF + GB.csv", "rb"), delimiter=",",
                      skiprows=2, usecols=[3, 4, 5])
euler_real = np.loadtxt(open(r"Subset1_syn.csv", "rb"), delimiter=",", skiprows=16,
                        usecols=[5, 6, 7])
data = np.loadtxt(open("Subset 1-Grain List.csv", "rb"), delimiter=",", skiprows=2,
                  usecols=[4, 5, 6, 7, 8, 9, 10, 11, 12, 3, 2])  # this is the data of original ipf map

# img = cv2.imread('kali_gen_sep.tif')
# img_voronoi = voronoi_optimization(img)
# img = cv2.imread('standard_size1.tif')
# img_cluster = cluster_img_dbscan(img)
# img_euler = voronoi_optimization_softwareDATA(data)

img0 = cv2.imread('standard_size1.tif')
imageshape = list(img0.shape[0:2])
img_euler = preprocessing_voronoi_euler(data) # optimize the microstructure using aniso voronoi, but the color is in euler angles, the ipf color is filled in generate file
