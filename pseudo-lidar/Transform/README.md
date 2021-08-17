# mcav
Monash Connected Autonomous vehicle perception package

Step 1.
------
Create a calibration directory (in current project) that would hold all calibration files. These calibration files must be formatted like shown in step 2 and 3 (for now, until we know how the matrices in from 
the street drone would look like).

In total we would have 3 matricies, i.e. the velo to cam rotation matrix (R, shape = 3x3), the velo to cam translation vector (T, shape = 3x1), and the cam to image projection matrix (P, shape = 3x4).

It would be your job to find these matrices (given your dataset or simulation env.) and structure them in each file.

Step 2. 
-------
Create a file called **calib_cam_to_cam.txt** in the calibration directory.
The contents of the file should be like below:

```
P: 1 2 3 4 5 6 7 8 9 10 11 12
```

where, P in matrix form would be:
```
P = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]]
```

Step 3. 
-------
Create a file called **calib_velo_to_cam.txt** in the calibration directory.
The contents of the file should be like below:

```
R: 1 2 3 4 5 6 7 8 9
T: 1 2 3
```

where, R and T in matrix form would be:
```
R = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

T = [1, 2, 3]
```

## Example
```python
from Transform.Transform import Transform as tf

# cloud is .bin for this example
cloud = (np.fromfile("<some cloud_dir>", dtype=np.float32)).reshape(-1, 4)
print(cloud.shape)

TF = tf("<calib directory>", img_width=1242, img_height=375)
depth_from_cloud = TF.project_velo_to_img(cloud)
print(depth_from_cloud.shape)
```