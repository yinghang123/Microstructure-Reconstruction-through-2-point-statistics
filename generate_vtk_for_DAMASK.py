import damask

dG = damask.Grid
#df = dG.load_DREAM3D(r'C:\media\lyh\Original_RVE_of_ti64.dream3d')
df = dG.load_ASCII(r'C:\media\export_from_dream3d\Ti64_syn_damaskfile1.geom')
print(df)
df.save('.\\generated_dataset\\realmesh_syn1', compress=False)