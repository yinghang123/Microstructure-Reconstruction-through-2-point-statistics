import codecs
import damask
import numpy as np

cm = damask.ConfigMaterial

######还是得从config和geom中获取
lines = []
with codecs.open(r'C:\media\export_from_dream3d\material.config', 'r', 'gb18030') as infile:
    for i in infile.readlines()[7:563:2]: ############这里需要查看material.config来改
        lines.append(i)
# print(lines)

eulers = [[] for i in range(len(lines))]
for i in range(len(lines)):
    a = lines[i].split()  # ['(gauss)', 'phi1', '114.385', 'Phi', '166.433', 'phi2', '48.992', 'scatter', '0.0', 'fraction', '1.0']
    eulers[i].append(float(a[2]))
    eulers[i].append(float(a[4]))
    eulers[i].append(float(a[6]))
    # print(a)
euler = np.array(eulers, dtype=np.float64)
eu = damask.Rotation.from_Euler_angles(euler, degrees=True)
print(eu)
my = cm.load(r'D:\media\scp_from_pai\material.yaml')
for i in range(len(eu)):
    my = my.material_add(O=eu[i], phase='Titanium', homogenization='SX')
print(my)
my.save('.\\generated_dataset\\material.yaml')
