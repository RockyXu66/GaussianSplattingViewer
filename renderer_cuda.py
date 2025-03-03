'''
Part of the code (CUDA and OpenGL memory transfer) is derived from https://github.com/jbaron34/torchwindow/tree/master
'''
from OpenGL import GL as gl
import OpenGL.GL.shaders as shaders
import util
import util_gau
import numpy as np
import torch
from renderer_ogl import GaussianRenderBase
from dataclasses import dataclass, field
from cuda import cudart as cu
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

try:
    from OpenGL.raw.WGL.EXT.swap_control import wglSwapIntervalEXT
except:
    wglSwapIntervalEXT = None


VERTEX_SHADER_SOURCE = """
#version 450

smooth out vec4 fragColor;
smooth out vec2 texcoords;

vec4 positions[3] = vec4[3](
    vec4(-1.0, 1.0, 0.0, 1.0),
    vec4(3.0, 1.0, 0.0, 1.0),
    vec4(-1.0, -3.0, 0.0, 1.0)
);

vec2 texpos[3] = vec2[3](
    vec2(0, 0),
    vec2(2, 0),
    vec2(0, 2)
);

void main() {
    gl_Position = positions[gl_VertexID];
    texcoords = texpos[gl_VertexID];
}
"""

FRAGMENT_SHADER_SOURCE = """
#version 330

smooth in vec2 texcoords;

out vec4 outputColour;

uniform sampler2D texSampler;

void main()
{
    outputColour = texture(texSampler, texcoords);
}
"""

@dataclass
class GaussianDataCUDA:
    xyz: torch.Tensor
    rot: torch.Tensor
    scale: torch.Tensor
    opacity: torch.Tensor
    sh: torch.Tensor = None
    precolor: torch.Tensor = None
    
    def __len__(self):
        return len(self.xyz)
    
    @property 
    def sh_dim(self):
        return self.sh.shape[-2]

@dataclass
class GaussianAvatarCUDA:
    num_points_per_subject:     torch.Tensor = None
    person_cnt_per_subject:     torch.Tensor = None
    rot:                        torch.Tensor = None
    scale:                      torch.Tensor = None
    opacity:                    torch.Tensor = None
    precolor:                   torch.Tensor = None
    xyz:                        torch.Tensor = None
    xyz_all:                    torch.Tensor = None
    sh:                         torch.Tensor = None
    query_lbs_list:             list = field(default_factory=list)
    cano_points_list:           list = field(default_factory=list)
    copies_list:                list = field(default_factory=list)
    

@dataclass
class GaussianRasterizationSettingsStorage:
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool


def gaus_cuda_from_cpu(gau: util_gau) -> GaussianDataCUDA:
    gaus =  GaussianDataCUDA(
        xyz = torch.tensor(gau.xyz).float().cuda().requires_grad_(False),
        rot = torch.tensor(gau.rot).float().cuda().requires_grad_(False),
        scale = torch.tensor(gau.scale).float().cuda().requires_grad_(False),
        opacity = torch.tensor(gau.opacity).float().cuda().requires_grad_(False),
        sh = torch.tensor(gau.sh).float().cuda().requires_grad_(False)
    )
    gaus.sh = gaus.sh.reshape(len(gaus), -1, 3).contiguous()
    return gaus

def avatar_cuda_from_cpu_raw(gau_list: list[util_gau.GaussianAvatarData]) -> GaussianAvatarCUDA:
    num_identities = len(gau_list)
    xyz_all = torch.concat([gau.xyz[0].repeat(gau.total_num_person, 1) for gau in gau_list], dim=0).float().cuda().requires_grad_(False)
    rot_all = torch.concat([gau.rot.repeat(gau.total_num_person, 1) for gau in gau_list], dim=0).float().cuda().requires_grad_(False)
    scale_all = torch.concat([gau.scale.repeat(gau.total_num_person, 1) for gau in gau_list], dim=0).float().cuda().requires_grad_(False)
    opacity_all = torch.concat([gau.opacity.repeat(gau.total_num_person, 1) for gau in gau_list], dim=0).float().cuda().requires_grad_(False)
    precolor_all = torch.concat([gau.colors_precomp.repeat(gau.total_num_person, 1) for gau in gau_list], dim=0).float().cuda().requires_grad_(False)

    ''' Set the position for the first frame from motions '''
    xyz_last_idx = 0
    for identity_idx in range(num_identities):
        copy_num_point_per_identity = gau_list[identity_idx].num_points_per_subject
        for copy in gau_list[identity_idx].copies:
            copy_num_person = copy['num_person']
            motion = copy['motion']
            transl_list = copy['transl_list']
            for person_idx in range(copy_num_person):
                cano2live_jnt_mats = motion[0:1].clone()  # only set the first frame
                pt_mats = torch.einsum('bnj,bjxy->bnxy', gau_list[identity_idx].query_lbs, cano2live_jnt_mats)
                xyz = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], gau_list[identity_idx].xyz) + pt_mats[..., :3, 3]
                xyz = xyz[0]
                xyz[:, 0:2] = -xyz[:, 0:2]
                xyz[:, :] += transl_list[person_idx]
                xyz_all[xyz_last_idx:xyz_last_idx+copy_num_point_per_identity] = xyz.cuda().requires_grad_(False)
                xyz_last_idx += copy_num_point_per_identity

    ''' For original cuda program '''
    num_points_concat = torch.empty((num_identities), dtype=torch.int32).cuda().requires_grad_(False)
    person_cnt_concat = torch.empty((num_identities), dtype=torch.int32).cuda().requires_grad_(False)
    gau_copies_list = []
    cano_points_list = []
    query_lbs_list = []
    for identity_idx in range(num_identities):
        for i in range(len(gau_list[identity_idx].copies)):
            gau_list[identity_idx].copies[i]['motion'] = gau_list[identity_idx].copies[i]['motion'].cuda()
            gau_list[identity_idx].copies[i]['transl_list'] = torch.tensor(np.array(gau_list[identity_idx].copies[i]['transl_list'])).cuda()
        copy_num_point_per_identity = gau_list[identity_idx].num_points_per_subject
        num_points_concat[identity_idx] = copy_num_point_per_identity
        person_cnt_concat[identity_idx] = gau_list[identity_idx].total_num_person
        gau_copies_list.append(gau_list[identity_idx].copies)
        cano_points_list.append(gau_list[identity_idx].xyz.cuda().requires_grad_(False))
        query_lbs_list.append(gau_list[identity_idx].query_lbs.cuda().requires_grad_(False))

    gau_gpu = GaussianAvatarCUDA(
        num_points_per_subject  = num_points_concat,
        person_cnt_per_subject  = person_cnt_concat,
        xyz                     = xyz_all,
        rot                     = rot_all,
        scale                   = scale_all,
        opacity                 = opacity_all,
        precolor                = precolor_all,
        query_lbs_list          = query_lbs_list,
        cano_points_list        = cano_points_list,
        copies_list             = gau_copies_list,
    )
    return gau_gpu

def avatar_cuda_from_cpu_optimized(gau_list: list[util_gau.GaussianAvatarData]) -> GaussianAvatarCUDA:
    num_identities = len(gau_list)
    total_num_gaus = np.sum([gau.total_num_person * gau.num_points_per_subject for gau in gau_list])

    ''' Collect xyz position data for all identites '''
    xyz_all = torch.zeros((total_num_gaus, 3), dtype=torch.float32).cuda().requires_grad_(False)
    num_points_concat = torch.empty((num_identities), dtype=torch.int32).cuda().requires_grad_(False)
    person_cnt_concat = torch.empty((num_identities), dtype=torch.int32).cuda().requires_grad_(False)
    xyz_last_idx = 0
    for identity_idx in range(num_identities):
        copy_num_point_per_identity = gau_list[identity_idx].num_points_per_subject
        num_points_concat[identity_idx] = copy_num_point_per_identity
        person_cnt_concat[identity_idx] = gau_list[identity_idx].total_num_person
        for copy in gau_list[identity_idx].copies:
            copy_num_person = copy['num_person']
            motion = copy['motion']
            transl_list = copy['transl_list']
            for person_idx in range(copy_num_person):
                cano2live_jnt_mats = motion[0:1].clone()  # only set the first frame
                pt_mats = torch.einsum('bnj,bjxy->bnxy', gau_list[identity_idx].query_lbs, cano2live_jnt_mats)
                xyz = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], gau_list[identity_idx].xyz) + pt_mats[..., :3, 3]
                xyz = xyz[0]
                xyz[:, 0:2] = -xyz[:, 0:2]
                xyz[:, :] += transl_list[person_idx]
                xyz_all[xyz_last_idx:xyz_last_idx+copy_num_point_per_identity] = xyz.cuda().requires_grad_(False)
                xyz_last_idx += copy_num_point_per_identity

    ''' Aggregate num_points_per_subject, motions, colors, scales, and query_lbs for different identities'''
    gau_copies_list = []
    cano_points_list = []
    query_lbs_list = []
    precolor_concat = torch.empty((torch.sum(num_points_concat), 3), dtype=torch.float32).cuda().requires_grad_(False)
    scale_concat = torch.empty((torch.sum(num_points_concat), 3), dtype=torch.float32).cuda().requires_grad_(False)
    last_idx = 0
    for identity_idx in range(num_identities):
        for i in range(len(gau_list[identity_idx].copies)):
            gau_list[identity_idx].copies[i]['motion'] = gau_list[identity_idx].copies[i]['motion'].cuda()
            gau_list[identity_idx].copies[i]['transl_list'] = torch.tensor(np.array(gau_list[identity_idx].copies[i]['transl_list'])).cuda()
        copy_num_point_per_identity = gau_list[identity_idx].num_points_per_subject
        gau_copies_list.append(gau_list[identity_idx].copies)
        cano_points_list.append(gau_list[identity_idx].xyz.cuda().requires_grad_(False))
        query_lbs_list.append(gau_list[identity_idx].query_lbs.cuda().requires_grad_(False))
        precolor_concat[last_idx:last_idx+copy_num_point_per_identity] = gau_list[identity_idx].colors_precomp.cuda().requires_grad_(False)
        scale_concat[last_idx:last_idx+copy_num_point_per_identity] = gau_list[identity_idx].scale.cuda().requires_grad_(False)
        last_idx += copy_num_point_per_identity
    gau_gpu = GaussianAvatarCUDA(
        num_points_per_subject  = num_points_concat,
        person_cnt_per_subject  = person_cnt_concat,
        xyz_all                 = xyz_all,
        scale                   = scale_concat,
        precolor                = precolor_concat,
        query_lbs_list          = query_lbs_list,
        cano_points_list        = cano_points_list,
        copies_list             = gau_copies_list,
    )
    return gau_gpu
   
import threading
import time
frame_idx = 0
total_frame_cnt = 0
animation_fps = 30
def update_frame_idx():
    global frame_idx
    while True:
        frame_idx += 1
        # if frame_idx >= total_frame_cnt:
        #     frame_idx = 0
        time.sleep(1 / animation_fps)  

class CUDARenderer(GaussianRenderBase):
    def __init__(self, w, h):
        super().__init__()
        self.raster_settings = {
            "image_height": int(h),
            "image_width": int(w),
            "tanfovx": 1,
            "tanfovy": 1,
            # "bg": torch.Tensor([0., 0., 0]).float().cuda(),
            "bg": torch.Tensor([1., 1., 1.]).float().cuda(),
            "scale_modifier": 1.,
            "viewmatrix": None,
            "projmatrix": None,
            "sh_degree": 3,  # ?
            "campos": None,
            "prefiltered": False,
            "debug": False
        }
        gl.glViewport(0, 0, w, h)
        self.program = util.compile_shaders(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
        # setup cuda
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise RuntimeError(
                "OpenGL context may be running on integrated graphics"
            )
        
        self.vao = gl.glGenVertexArrays(1)
        self.tex = None
        self.set_gl_texture(h, w)

        gl.glDisable(gl.GL_CULL_FACE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.need_rerender = True
        self.update_vsync()

        self.motion_data = None

        # Start the frame index update thread
        frame_thread = threading.Thread(target=update_frame_idx)
        frame_thread.daemon = True
        frame_thread.start()

    def update_vsync(self):
        if wglSwapIntervalEXT is not None:
            wglSwapIntervalEXT(1 if self.reduce_updates else 0)
        # else:
        #     print("VSync is not supported")

    def update_gaussian_data(self, gaus: util_gau.GaussianData):
        self.need_rerender = True
        self.gaussians = gaus_cuda_from_cpu(gaus)
        self.raster_settings["sh_degree"] = int(np.round(np.sqrt(self.gaussians.sh_dim))) - 1

    def update_gaussian_avatar_w_precolor(self, gaus_list: list[util_gau.GaussianAvatarData], optimized:bool = False):
        self.need_rerender = True
        if optimized:
            self.gaussians = avatar_cuda_from_cpu_optimized(gaus_list)
        else:
            self.gaussians = avatar_cuda_from_cpu_raw(gaus_list)
    
    def sort_and_update(self, camera: util.Camera):
        self.need_rerender = True

    def set_scale_modifier(self, modifier):
        self.need_rerender = True
        self.raster_settings["scale_modifier"] = float(modifier)

    def set_render_mod(self, mod: int):
        self.need_rerender = True

    def set_gl_texture(self, h, w):
        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            w,
            h,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to register opengl texture")
    
    def set_render_reso(self, w, h):
        self.need_rerender = True
        self.raster_settings["image_height"] = int(h)
        self.raster_settings["image_width"] = int(w)
        gl.glViewport(0, 0, w, h)
        self.set_gl_texture(h, w)

    def update_camera_pose(self, camera: util.Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["viewmatrix"] = torch.tensor(view_matrix.T).float().cuda()
        self.raster_settings["campos"] = torch.tensor(camera.position).float().cuda()
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()

    def update_camera_intrin(self, camera: util.Camera):
        self.need_rerender = True
        view_matrix = camera.get_view_matrix()
        view_matrix[[0, 2], :] = -view_matrix[[0, 2], :]
        proj = camera.get_project_matrix() @ view_matrix
        self.raster_settings["projmatrix"] = torch.tensor(proj.T).float().cuda()
        hfovx, hfovy, focal = camera.get_htanfovxy_focal()
        self.raster_settings["tanfovx"] = hfovx
        self.raster_settings["tanfovy"] = hfovy

    def draw(self):
        if self.reduce_updates and not self.need_rerender:
            gl.glUseProgram(self.program)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            return

        self.need_rerender = False

        # run cuda rasterizer now is just a placeholder
        # img = torch.meshgrid((torch.linspace(0, 1, 720), torch.linspace(0, 1, 1280)))
        # img = torch.stack([img[0], img[1], img[1], img[1]], dim=-1)
        # img = img.float().cuda(0)
        # img = img.contiguous()
        raster_settings = GaussianRasterizationSettings(**self.raster_settings)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        # means2D = torch.zeros_like(self.gaussians.xyz, dtype=self.gaussians.xyz.dtype, requires_grad=False, device="cuda")
        with torch.no_grad():
            img, radii = rasterizer(
                means3D = self.gaussians.xyz,
                means2D = None,
                shs = self.gaussians.sh,
                colors_precomp = None,
                opacities = self.gaussians.opacity,
                scales = self.gaussians.scale,
                rotations = self.gaussians.rot,
                cov3D_precomp = None
            )

        img = img.permute(1, 2, 0)
        img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
        img = img.contiguous()
        height, width = img.shape[:2]
        # transfer
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to map graphics resource")
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to get mapped array")
        
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            img.data_ptr(),
            4 * 4 * width,
            4 * 4 * width,
            height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to copy from tensor to texture")
        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to unmap graphics resource")

        gl.glUseProgram(self.program)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

    def update_pos(self, optimized=False):

        num_identities = len(self.gaussians.cano_points_list)
        xyz_last_idx = 0
        for identity_idx in range(num_identities):
            copy_num_point_per_identity = self.gaussians.num_points_per_subject[identity_idx]
            copies = self.gaussians.copies_list[identity_idx]
            query_lbs = self.gaussians.query_lbs_list[identity_idx]
            for copy in copies:
                motion_len = copy['motion'].shape[0]
                num_person = copy['num_person']

                cano_deform_point = self.gaussians.cano_points_list[identity_idx].repeat(num_person, 1, 1)
                cano2live_jnt_mats = copy['motion'][frame_idx%motion_len : frame_idx%motion_len+1].clone().repeat(num_person, 1, 1, 1)
                pt_mats = torch.einsum('bnj,bjxy->bnxy', query_lbs, cano2live_jnt_mats)
                xyz = torch.einsum('bnxy,bny->bnx', pt_mats[..., :3, :3], cano_deform_point) + pt_mats[..., :3, 3]
                xyz[:, :2, :] = 0.0
                xyz[:, :, 0:2] = -xyz[:, :, 0:2]
                xyz[:, :, :] += copy['transl_list'].unsqueeze(1).repeat(1, xyz.shape[1], 1)
                xyz = xyz.view(-1, 3)
                if optimized:
                    self.gaussians.xyz_all[xyz_last_idx:xyz_last_idx+copy_num_point_per_identity*num_person] = xyz
                else:
                    self.gaussians.xyz[xyz_last_idx:xyz_last_idx+copy_num_point_per_identity*num_person] = xyz
                xyz_last_idx += copy_num_point_per_identity*num_person

    def draw_w_precolor(self, optimized=False):
        if self.reduce_updates and not self.need_rerender:
            gl.glUseProgram(self.program)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
            return

        # self.need_rerender = False

        # run cuda rasterizer now is just a placeholder
        # img = torch.meshgrid((torch.linspace(0, 1, 720), torch.linspace(0, 1, 1280)))
        # img = torch.stack([img[0], img[1], img[1], img[1]], dim=-1)
        # img = img.float().cuda(0)
        # img = img.contiguous()
        raster_settings = GaussianRasterizationSettings(**self.raster_settings)
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        # means2D = torch.zeros_like(self.gaussians.xyz, dtype=self.gaussians.xyz.dtype, requires_grad=False, device="cuda")

        if optimized:
            with torch.no_grad():
                img, radii = rasterizer(
                    num_identities = self.gaussians.num_points_per_subject.shape[0],
                    num_points_per_subject = self.gaussians.num_points_per_subject,
                    person_cnt_per_subject = self.gaussians.person_cnt_per_subject,
                    means3D = self.gaussians.xyz_all,
                    means2D = None,
                    shs = None,
                    colors_precomp = self.gaussians.precolor,
                    scales = self.gaussians.scale,
                    cov3D_precomp = None
                )
        else:
            with torch.no_grad():
                img, radii = rasterizer(
                    means3D = self.gaussians.xyz,
                    means2D = None,
                    shs = None,
                    colors_precomp = self.gaussians.precolor,
                    opacities = self.gaussians.opacity,
                    scales = self.gaussians.scale,
                    rotations = self.gaussians.rot,
                    cov3D_precomp = None
                )

        img = img.permute(1, 2, 0)
        img = torch.concat([img, torch.ones_like(img[..., :1])], dim=-1)
        img = img.contiguous()
        height, width = img.shape[:2]

        """
        import cv2
        import os
        os.makedirs('images', exist_ok=True)
        cv2.imwrite(f'images/img_{frame_idx:06d}.png', cv2.cvtColor(img.detach().cpu().numpy()*255, cv2.COLOR_BGR2RGB))
        """
        # if frame_idx % 450 == 0:
        #     import cv2
        #     import os
        #     os.makedirs('images', exist_ok=True)
        #     cv2.imwrite(f'images/img_{frame_idx:06d}.png', cv2.cvtColor(img.detach().cpu().numpy()*255, cv2.COLOR_BGR2RGB))

        # transfer
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to map graphics resource")
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to get mapped array")
        
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            img.data_ptr(),
            4 * 4 * width,
            4 * 4 * width,
            height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to copy from tensor to texture")
        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise RuntimeError("Unable to unmap graphics resource")

        gl.glUseProgram(self.program)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)