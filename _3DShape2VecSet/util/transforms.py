import numpy as np

class RandomMirrorRotate3D:
    """
    In-place mirror + random ±angle° rotations on 3D point clouds.
    pts and surf are both (N,3) numpy arrays in [-1,1]^3.
    """

    def __init__(self, max_angle=10.0, mirror_prob=0.5):
        self.max_angle   = np.deg2rad(max_angle)
        self.mirror_prob = mirror_prob

    def __call__(self, pts, surf, normals):
        # print("applying transform")
        # 1) Mirror along x-axis (left <-> right)
        if np.random.rand() < self.mirror_prob:
            pts[:,0]  = -pts[:,0]
            surf[:,0] = -surf[:,0]
            normals[:,0] = -normals[:,0]

        # 2) Random rotations about X, Y, Z
        #    sample three angles in [-max_angle, max_angle]
        angles = (np.random.rand(3)*2 - 1) * self.max_angle
        cx, sx = np.cos(angles[0]), np.sin(angles[0])
        cy, sy = np.cos(angles[1]), np.sin(angles[1])
        cz, sz = np.cos(angles[2]), np.sin(angles[2])

        # rotation matrices
        Rx = np.array([[1,0,0],
                       [0, cx,-sx],
                       [0, sx, cx]], dtype=np.float32)
        Ry = np.array([[cy,0,sy],
                       [0, 1, 0],
                       [-sy,0,cy]], dtype=np.float32)
        Rz = np.array([[cz,-sz,0],
                       [sz, cz,0],
                       [0,   0,1]], dtype=np.float32)

        R = Rz @ Ry @ Rx    # first X, then Y, then Z

        # apply rotation
        pts[:]  = pts.dot(R.T)
        surf[:] = surf.dot(R.T)
        normals = normals.dot(R.T)

        return pts, surf, normals
