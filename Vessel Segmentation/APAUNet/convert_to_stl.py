#THIS FILE IS A PREPROCESSING STEP TO SHOW THE 3D MODEL IN QML OR IN JAVASCRIPT
import nibabel as nib
import numpy as np
from stl import mesh
from skimage import measure

file_path = "C:/Users/Jorge/Desktop/outputs_batch10_epoch_212.nii.gz"

nifti_file = nib.load(file_path)
np_array = nifti_file.get_fdata()

verts, faces, normals, values = measure.marching_cubes(np_array, 0) # type: ignore

obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

for i, f in enumerate(faces):
    obj_3d.vectors[i] = verts[f]

obj_3d.save('outputs_batch10_epoch_212.stl')