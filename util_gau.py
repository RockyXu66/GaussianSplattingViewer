import numpy as np
from plyfile import PlyData
from dataclasses import dataclass
from loguru import logger
import torch
import random

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray = None
    colors_precomp: np.ndarray = None
    query_lbs: torch.Tensor = None
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
    query_lbs: torch.Tensor = None
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

def load_pt(file_path):
    data = torch.load(file_path)
    xyz = data['position'].unsqueeze(0)
    # xyz = rotate_point_cloud(xyz)
    xyz[:, 0:2] = -xyz[:, 0:2]
    rot = data['rotation']
    scale = data['scale']
    opacity = data['opacity']
    colors_precomp = data['color']
    query_lbs = data['query_lbs'].unsqueeze(0)
    return GaussianData(xyz=xyz, rot=rot, scale=scale, opacity=opacity, colors_precomp=colors_precomp, query_lbs=query_lbs)

def load_motion(file_path):
    xyz_list = np.load(file_path)
    xyz_list[:, :, 0:2] = -xyz_list[:, :, 0:2]
    # for i in range(xyz_list.shape[0]):
    #     xyz_list[i] = rotate_point_cloud(xyz_list[i])
    return xyz_list

def set_transl(total_num_person, row, col, unit_dist=1.3, shuffle_sequence=True):
    assert row * col >= total_num_person, f'total number of persons ({total_num_person}) is bigger than row x col ({row * col})'
    transl_list = []
    for row_idx in range(row):
        for col_idx in range(col):
            noise = np.random.normal(0, 0.3, size=2)
            transl_list.append(np.array([(col_idx - col / 2) * unit_dist + noise[0], 0, -row_idx * unit_dist + noise[1]]))
    transl_list = np.array(transl_list)
    if shuffle_sequence:
        np.random.shuffle(transl_list)
    return transl_list

def load_identity(identity_dict, transl_list, transl_idx):
    file_path = identity_dict['id']
    data = load_pt(file_path)

    copies = []
    total_num_person = 0
    num_points_per_subject = data.rot.shape[0]
    for copy in identity_dict['copy']:
        motion_file_path = copy['motion']
        motion_data = torch.load(motion_file_path)
        num_person = copy['num_person']
        identity_transl_list = []
        for _ in range(num_person):
            # motion_list.append(random_rotate(motion_data) + np.array([j - col/2, 0, i]))
            # motion_list[i * col + j] = random_rotate(motion_data) + grid_pos[i, j]
            identity_transl_list.append(transl_list[transl_idx])
            transl_idx += 1
        copies.append({
            'num_person': num_person,
            'motion': motion_data,
            'transl_list': identity_transl_list,
        })
        total_num_person += num_person

    gau_avatar_data = GaussianAvatarData(
        xyz=data.xyz, rot=data.rot, scale=data.scale, 
        opacity=data.opacity, colors_precomp=data.colors_precomp,
        query_lbs=data.query_lbs,
        copies=copies,
        total_num_person=total_num_person,
        num_points_per_subject=num_points_per_subject,
    )
    return gau_avatar_data, transl_idx

def load_crowd(crowd_list, row, col, unit_dist=1.3, shuffle_sequence=True, cam_position=None):
    logger.info(f'Grid size: row x col -> {row} x {col}')
    total_num_person = np.sum([copy['num_person'] for identity_dict in crowd_list for copy in identity_dict['copy']])
    transl_list = set_transl(total_num_person, row, col, unit_dist=1.3, shuffle_sequence=shuffle_sequence)
    gau_avatar_list = []
    transl_idx = 0
    for identity in crowd_list:
        gau_avatar, transl_idx = load_identity(identity, transl_list, transl_idx)
        gau_avatar_list.append(gau_avatar)
    return gau_avatar_list

def divide_into_parts(A, B):
    parts = np.zeros(B, dtype=int)
    base_value = A // B
    parts[:] = base_value
    remainder = A % B
    parts[:remainder] += 1
    return parts

def load_crowd_grid(row, col, unit_dist=1.3, shuffle_sequence=True, cam_position=None, motion_list=None, identity_list=None):
    logger.info(f'Grid size: row x col -> {row} x {col}')
    total_num_person = row * col
    transl_list = set_transl(total_num_person, row, col, unit_dist=unit_dist, shuffle_sequence=shuffle_sequence)

    LOD_0 = [transl for transl in transl_list if np.linalg.norm(transl-cam_position) <= 5]
    LOD_1 = [transl for transl in transl_list if 5 < np.linalg.norm(transl-cam_position) <= 10]
    LOD_2 = [transl for transl in transl_list if 10 < np.linalg.norm(transl-cam_position)]

    gau_avatar_list = []

    transl_idx = 0
    LOD_0_identity_list = identity_list['res_512']
    # LOD_0_identity_list = np.random.choice(LOD_0_identity_list, len(LOD_0))
    assert len(LOD_0_identity_list) <= len(LOD_0)
    # LOD_0_identity_total_num_person_list = np.arange(1, len(LOD_0)+1, int(len(LOD_0)/len(LOD_0_identity_list)))
    LOD_0_identity_total_num_person_list = divide_into_parts(len(LOD_0), len(LOD_0_identity_list))
    for idx in range(len(LOD_0_identity_list)):
        identity_name = LOD_0_identity_list[idx]
        identity_total_num_person = LOD_0_identity_total_num_person_list[idx]
        num_copies = 1
        LOD_0_motion_list = np.random.choice(motion_list, num_copies)
        # num_person_list = np.arange(1, identity_total_num_person+1, num_copies)
        assert identity_total_num_person >= num_copies
        num_person_list = divide_into_parts(identity_total_num_person, num_copies)
        copies = []
        for copy_idx in range(num_copies):
            num_person = num_person_list[copy_idx]
            copies.append({
                'num_person': num_person,
                'motion': LOD_0_motion_list[copy_idx],
            })
        identity = {
            'id': identity_name,
            'copy': copies,
        }
        gau_avatar, transl_idx = load_identity(identity, LOD_0, transl_idx)
        gau_avatar_list.append(gau_avatar)

    transl_idx = 0
    LOD_1_identity_list = identity_list['res_128']
    assert len(LOD_1_identity_list) <= len(LOD_1)
    LOD_1_identity_total_num_person_list = divide_into_parts(len(LOD_1), len(LOD_1_identity_list))
    for idx in range(len(LOD_1_identity_list)):
        identity_name = LOD_1_identity_list[idx]
        identity_total_num_person = LOD_1_identity_total_num_person_list[idx]
        num_copies = 4
        LOD_1_motion_list = np.random.choice(motion_list, num_copies)
        assert identity_total_num_person >= num_copies
        num_person_list = divide_into_parts(identity_total_num_person, num_copies)
        copies = []
        for copy_idx in range(num_copies):
            num_person = num_person_list[copy_idx]
            copies.append({
                'num_person': num_person,
                'motion': LOD_1_motion_list[copy_idx],
            })
        identity = {
            'id': identity_name,
            'copy': copies,
        }
        gau_avatar, transl_idx = load_identity(identity, LOD_1, transl_idx)
        gau_avatar_list.append(gau_avatar)

    transl_idx = 0
    LOD_2_identity_list = identity_list['res_64']
    assert len(LOD_2_identity_list) <= len(LOD_2)
    LOD_2_identity_total_num_person_list = divide_into_parts(len(LOD_2), len(LOD_2_identity_list))
    for idx in range(len(LOD_2_identity_list)):
        identity_name = LOD_2_identity_list[idx]
        identity_total_num_person = LOD_2_identity_total_num_person_list[idx]
        num_copies = 4
        LOD_2_motion_list = np.random.choice(motion_list, num_copies)
        assert identity_total_num_person >= num_copies
        num_person_list = divide_into_parts(identity_total_num_person, num_copies)
        copies = []
        for copy_idx in range(num_copies):
            num_person = num_person_list[copy_idx]
            copies.append({
                'num_person': num_person,
                'motion': LOD_2_motion_list[copy_idx],
            })
        identity = {
            'id': identity_name,
            'copy': copies,
        }
        gau_avatar, transl_idx = load_identity(identity, LOD_2, transl_idx)
        gau_avatar_list.append(gau_avatar)
    
    return gau_avatar_list

if __name__ == "__main__":
    gs = load_ply("C:\\Users\\MSI_NB\\Downloads\\viewers\\models\\train\\point_cloud\\iteration_7000\\point_cloud.ply")
    a = gs.flat()
    print(a.shape)
