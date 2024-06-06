import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
from loguru import logger
import random

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray = None
    colors_precomp: np.ndarray = None
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-1]

@dataclass
class GaussianAvatarData:
    total_num_person: int
    num_points_per_subject: int
    copies: list
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray = None
    colors_precomp: np.ndarray = None
    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)
    
    def __len__(self):
        return len(self.xyz)


def naive_gaussian():
    gau_xyz = np.array([
        0, 0, 0,
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    ]).astype(np.float32).reshape(-1, 3)
    gau_rot = np.array([
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0,
        1, 0, 0, 0
    ]).astype(np.float32).reshape(-1, 4)
    gau_s = np.array([
        0.03, 0.03, 0.03,
        0.2, 0.03, 0.03,
        0.03, 0.2, 0.03,
        0.03, 0.03, 0.2
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = np.array([
        1, 0, 1, 
        1, 0, 0, 
        0, 1, 0, 
        0, 0, 1, 
    ]).astype(np.float32).reshape(-1, 3)
    gau_c = (gau_c - 0.5) / 0.28209
    gau_a = np.array([
        1, 1, 1, 1
    ]).astype(np.float32).reshape(-1, 1)
    return GaussianData(
        gau_xyz,
        gau_rot,
        gau_s,
        gau_a,
        gau_c
    )


def load_ply(path):
    max_sh_degree = 3
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    scales = np.exp(scales)
    scales = scales.astype(np.float32)
    opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([features_dc.reshape(-1, 3), 
                        features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    return GaussianData(xyz, rots, scales, opacities, shs)


def load_npz(file_path):
    '''
    Save the npz model in `/raid/yixu/Projects/GaussianSplatting/GaussianAvatar/model/avatar_model.py (850)`
    '''
    data = np.load(file_path)
    xyz = data['means3D']
    # xyz = rotate_point_cloud(xyz)
    xyz[:, 0:2] = -xyz[:, 0:2]
    rot = data['rotations']
    scale = data['scales']
    opacity = data['opacities']
    colors_precomp = data['colors_precomp']
    return GaussianData(xyz=xyz, rot=rot, scale=scale, opacity=opacity, colors_precomp=colors_precomp)

def load_motion(file_path):
    xyz_list = np.load(file_path)
    xyz_list[:, :, 0:2] = -xyz_list[:, :, 0:2]
    # for i in range(xyz_list.shape[0]):
    #     xyz_list[i] = rotate_point_cloud(xyz_list[i])
    return xyz_list

def load_identity(identity_dict):
    file_path = identity_dict['id']
    data = load_npz(file_path)

    copies = []
    total_num_person = 0
    radius_range = 10
    num_points_per_subject = data.rot.shape[0]
    row = 10; col = 10      # 100
    # row = 35; col = 30      # 1000
    # row = 45; col = 45      # 2000
    # row = 71; col = 71      # 5000
    logger.info(f'Grid size: row x col -> {row} x {col}')
    for copy in identity_dict['copy']:
        motion_file_path = copy['motion']
        motion_data = load_motion(motion_file_path)
        num_person = copy['num_person']
        transl_list = []
        # motion_list = np.empty((num_person, motion_data.shape[0], num_points_per_subject, 3), dtype=np.float32)
        # motion_list = []
        unit_dist = 1.3
        assert row * col > total_num_person and row * col > num_person
        # grid_pos = np.arange(0, row * col).reshape(row, col)
        for i in range(row):
            for j in range(col):
                # motion_list.append(random_rotate(motion_data) + np.array([j - col/2, 0, i]))
                # motion_list[i * col + j] = random_rotate(motion_data) + grid_pos[i, j]
                transl_list.append(np.array([(j - col/2)*unit_dist, 0, -i*unit_dist]))
                # motion_list.append(motion_data)
                if len(transl_list) == num_person:
                    break
            if len(transl_list) == num_person:
                break
        copies.append({
            'num_person': num_person,
            'motion': motion_data,
            'transl_list': transl_list,
        })
        total_num_person += num_person

    return GaussianAvatarData(
        xyz=data.xyz, rot=data.rot, scale=data.scale, 
        opacity=data.opacity, colors_precomp=data.colors_precomp,
        copies=copies,
        total_num_person=total_num_person,
        num_points_per_subject=num_points_per_subject,
    )

if __name__ == "__main__":
    gs = load_ply("C:\\Users\\MSI_NB\\Downloads\\viewers\\models\\train\\point_cloud\\iteration_7000\\point_cloud.ply")
    a = gs.flat()
    print(a.shape)
