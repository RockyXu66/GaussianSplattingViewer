import glfw
import OpenGL.GL as gl
from imgui.integrations.glfw import GlfwRenderer
import imgui
import numpy as np
import util
import imageio
import util_gau
import tkinter as tk
from tkinter import filedialog
import os
import sys
import argparse
from renderer_ogl import OpenGLRenderer, GaussianRenderBase
from renderer_cuda import CUDARenderer


# Add the directory containing main.py to the Python path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

# Change the current working directory to the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

cam_position = np.array([0.0, -0.0, 3.0]).astype(np.float32)
target = np.array([0.0, -0.2, 0.0]).astype(np.float32)
up = np.array([0.0, -1.0, 0.0]).astype(np.float32)

g_camera = util.Camera(720, 1280, cam_position, target, up)
BACKEND_OGL=1
BACKEND_CUDA=0
g_renderer_list = [
    None, # ogl
]
g_renderer_idx = BACKEND_CUDA
g_renderer: CUDARenderer = g_renderer_list[g_renderer_idx]
g_scale_modifier = 1.
g_auto_sort = True
g_show_control_win = True
g_show_help_win = True
g_show_camera_win = False
g_render_mode_tables = ["Gaussian Ball", "Flat Ball", "Billboard", "Depth", "SH:0", "SH:0~1", "SH:0~2", "SH:0~3 (default)"]
g_render_mode = 7

def impl_glfw_init():
    window_name = "NeUVF editor"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    # glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    global window
    window = glfw.create_window(
        g_camera.w, g_camera.h, window_name, None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(0)
    # glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_NORMAL);
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        exit(1)

    return window

def cursor_pos_callback(window, xpos, ypos):
    if imgui.get_io().want_capture_mouse:
        g_camera.is_leftmouse_pressed = False
        g_camera.is_rightmouse_pressed = False
    g_camera.process_mouse(xpos, ypos)

def mouse_button_callback(window, button, action, mod):
    if imgui.get_io().want_capture_mouse:
        return
    pressed = action == glfw.PRESS
    g_camera.is_leftmouse_pressed = (button == glfw.MOUSE_BUTTON_LEFT and pressed)
    g_camera.is_rightmouse_pressed = (button == glfw.MOUSE_BUTTON_RIGHT and pressed)

def wheel_callback(window, dx, dy):
    g_camera.process_wheel(dx, dy)

def key_callback(window, key, scancode, action, mods):
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_Q:
            g_camera.process_roll_key(1)
        elif key == glfw.KEY_E:
            g_camera.process_roll_key(-1)

def update_camera_pose_lazy():
    if g_camera.is_pose_dirty:
        g_renderer.update_camera_pose(g_camera)
        g_camera.is_pose_dirty = False

def update_camera_intrin_lazy():
    if g_camera.is_intrin_dirty:
        g_renderer.update_camera_intrin(g_camera)
        g_camera.is_intrin_dirty = False

def update_activated_renderer_state(gaus: util_gau.GaussianData):
    g_renderer.update_gaussian_data(gaus)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def update_activated_renderer_state_avatar(gaus_list: list[util_gau.GaussianAvatarData], optimized=False):
    g_renderer.update_gaussian_avatar_w_precolor(gaus_list, optimized)
    g_renderer.sort_and_update(g_camera)
    g_renderer.set_scale_modifier(g_scale_modifier)
    g_renderer.set_render_mod(g_render_mode - 3)
    g_renderer.update_camera_pose(g_camera)
    g_renderer.update_camera_intrin(g_camera)
    g_renderer.set_render_reso(g_camera.w, g_camera.h)

def window_resize_callback(window, width, height):
    gl.glViewport(0, 0, width, height)
    g_camera.update_resolution(height, width)
    g_renderer.set_render_reso(width, height)

def main():
    global g_camera, g_renderer, g_renderer_list, g_renderer_idx, g_scale_modifier, g_auto_sort, \
        g_show_control_win, g_show_help_win, g_show_camera_win, \
        g_render_mode, g_render_mode_tables
        
    imgui.create_context()
    if args.hidpi:
        imgui.get_io().font_global_scale = 1.5
    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    # Load a custom font
    io = imgui.get_io()
    font_path = "Roboto/Roboto-VariableFont_wdth,wght.ttf"  # Replace with the actual font file path
    custom_font = io.fonts.add_font_from_file_ttf(font_path, 20)  # Adjust the size as needed
    impl.refresh_font_texture()  # Apply the font

    root = tk.Tk()  # used for file dialog
    root.withdraw()
    
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_scroll_callback(window, wheel_callback)
    glfw.set_key_callback(window, key_callback)
    
    glfw.set_window_size_callback(window, window_resize_callback)

    # init renderer
    g_renderer_list[BACKEND_CUDA] = CUDARenderer(g_camera.w, g_camera.h)
    g_renderer_list += [OpenGLRenderer(g_camera.w, g_camera.h)]
    g_renderer = g_renderer_list[g_renderer_idx]

    # gaussian data
    gaussians = util_gau.naive_gaussian()

    model_root_folder = 'gs-crowd-avatars/models'
    motion_root_folder = 'gs-crowd-avatars/motions'

    identity_list = {
        'res_512': [
            f'{model_root_folder}/Nivesh-2406141613_512_stage1-res.pt',
            f'{model_root_folder}/Theo_IMG_3783_512_stage1-res.pt',
        ],
        'res_128': [
            f'{model_root_folder}/Nivesh-2406141613_128_stage1-res.pt',
            f'{model_root_folder}/Theo_IMG_3783_128_stage1-res.pt',
        ],
        'res_64': [
            f'{model_root_folder}/Nivesh-2406141613_64_stage1-res.pt',
            f'{model_root_folder}/Theo_IMG_3783_64_stage1-res.pt',
        ],
    }
    motion_list = [
        
        'Extended 3_poses.pt',
        'Walk B22 - Side step left_poses.pt',
    ]
    motion_list = [f'{motion_root_folder}/{motion}' for motion in motion_list]

    crowd_list = [
        {
            'id': f'{model_root_folder}/Theo_0522_1624_512_stage1_20240522_215411/Theo_0522_1624_512_stage1_20240522_215411-res.pt',
            'copy': [ 
                {'num_person': 100, 'motion': f'{motion_root_folder}/Andria_Satisfied_v1_C3D_poses.pt'}, 
            ]
        },
        # {
        #     'id': f'{model_root_folder}/Theo_0523_1707_64_stage1_20240611_114939/Theo_0523_1707_64_stage1_20240611_114939-res.pt',
        #     'copy': [ 
        #         {'num_person': 100, 'motion': f'{motion_root_folder}/Vasso_Bachata_01_poses.pt'}, 
        #     ]
        # },
    ]
    optimized = True
    # optimized = False
    with_motion = True
    # with_motion = False

    # row = 1; col = 1;
    # row = 3; col = 3        # 100
    # row = 10; col = 10        # 100
    # row = 20; col = 20;       # 400
    # row = 35; col = 30        # 1000
    row = 45; col = 45        # 2000
    # row = 70; col = 50        # 3500
    # row = 71; col = 71        # 5000
    unit_dist = 1.3             # distance between two people
    shuffle_sequence = True     # wheather to shuffle the sequence

    # Use given crowd_list
    # gau_avatar_list = util_gau.load_crowd(crowd_list, row, col, unit_dist=unit_dist, shuffle_sequence=shuffle_sequence, cam_position=cam_position)

    # Randomly chose the identity and the motion
    gau_avatar_list = util_gau.load_crowd_grid(row, col, unit_dist=unit_dist, shuffle_sequence=shuffle_sequence, motion_list=motion_list, identity_list=identity_list, cam_position=cam_position)
    update_activated_renderer_state_avatar(gau_avatar_list, optimized)
    
    # settings
    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        
        gl.glClearColor(0, 0, 0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        update_camera_pose_lazy()
        update_camera_intrin_lazy()
        
        # g_renderer.draw()
        if with_motion:
            g_renderer.update_pos(optimized)
        g_renderer.draw_w_precolor(optimized)

        imgui.push_font(custom_font)

        # imgui ui
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("Window", True):
                clicked, g_show_control_win = imgui.menu_item(
                    "Show Control", None, g_show_control_win
                )
                clicked, g_show_help_win = imgui.menu_item(
                    "Show Help", None, g_show_help_win
                )
                clicked, g_show_camera_win = imgui.menu_item(
                    "Show Camera Control", None, g_show_camera_win
                )
                imgui.end_menu()
            imgui.end_main_menu_bar()
        
        if g_show_control_win:
            if imgui.begin("Control", True):
                # rendering backend
                changed, g_renderer_idx = imgui.combo("backend", g_renderer_idx, ["cuda", "ogl"][:len(g_renderer_list)])
                if changed:
                    g_renderer = g_renderer_list[g_renderer_idx]
                    update_activated_renderer_state(gaussians)

                imgui.text(f"fps = {imgui.get_io().framerate:.1f}")

                # changed, g_renderer.reduce_updates = imgui.checkbox(
                #         "reduce updates", g_renderer.reduce_updates,
                #     )

                total_num_gaus = np.sum([gau_avatar.total_num_person * gau_avatar.num_points_per_subject for gau_avatar in gau_avatar_list])
                # imgui.text(f"# of Gaus = {total_num_gaus}")
                imgui.text(f"# characters = {col * row}")
                # if imgui.button(label='open ply'):
                #     file_path = filedialog.askopenfilename(title="open ply",
                #         initialdir="C:\\Users\\MSI_NB\\Downloads\\viewers",
                #         filetypes=[('ply file', '.ply')]
                #         )
                #     if file_path:
                #         try:
                #             gaussians = util_gau.load_ply(file_path)
                #             g_renderer.update_gaussian_data(gaussians)
                #             g_renderer.sort_and_update(g_camera)
                #         except RuntimeError as e:
                #             pass
                
                # camera fov
                changed, g_camera.fovy = imgui.slider_float(
                    "fov", g_camera.fovy, 0.001, np.pi - 0.001, "fov = %.3f"
                )
                g_camera.is_intrin_dirty = changed
                update_camera_intrin_lazy()
                
                # # scale modifier
                # changed, g_scale_modifier = imgui.slider_float(
                #     "", g_scale_modifier, 0.1, 10, "scale modifier = %.3f"
                # )
                # imgui.same_line()
                # if imgui.button(label="reset"):
                #     g_scale_modifier = 1.
                #     changed = True
                    
                # if changed:
                #     g_renderer.set_scale_modifier(g_scale_modifier)
                
                # # render mode
                # changed, g_render_mode = imgui.combo("shading", g_render_mode, g_render_mode_tables)
                # if changed:
                #     g_renderer.set_render_mod(g_render_mode - 4)
                
                # # sort button
                # if imgui.button(label='sort Gaussians'):
                #     g_renderer.sort_and_update(g_camera)
                # imgui.same_line()
                # changed, g_auto_sort = imgui.checkbox(
                #         "auto sort", g_auto_sort,
                #     )
                # if g_auto_sort:
                #     g_renderer.sort_and_update(g_camera)
                
                # if imgui.button(label='save image'):
                #     width, height = glfw.get_framebuffer_size(window)
                #     nrChannels = 3;
                #     stride = nrChannels * width;
                #     stride += (4 - stride % 4) if stride % 4 else 0
                #     gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
                #     gl.glReadBuffer(gl.GL_FRONT)
                #     bufferdata = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                #     img = np.frombuffer(bufferdata, np.uint8, -1).reshape(height, width, 3)
                #     imageio.imwrite("save.png", img[::-1])
                #     # save intermediate information
                #     # np.savez(
                #     #     "save.npz",
                #     #     gau_xyz=gaussians.xyz,
                #     #     gau_s=gaussians.scale,
                #     #     gau_rot=gaussians.rot,
                #     #     gau_c=gaussians.sh,
                #     #     gau_a=gaussians.opacity,
                #     #     viewmat=g_camera.get_view_matrix(),
                #     #     projmat=g_camera.get_project_matrix(),
                #     #     hfovxyfocal=g_camera.get_htanfovxy_focal()
                #     # )
                imgui.end()

        if g_show_camera_win:
            if imgui.button(label='rot 180'):
                g_camera.flip_ground()

            changed, g_camera.target_dist = imgui.slider_float(
                    "t", g_camera.target_dist, 1., 8., "target dist = %.3f"
                )
            if changed:
                g_camera.update_target_distance()

            changed, g_camera.rot_sensitivity = imgui.slider_float(
                    "r", g_camera.rot_sensitivity, 0.002, 0.1, "rotate speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset r"):
                g_camera.rot_sensitivity = 0.02

            changed, g_camera.trans_sensitivity = imgui.slider_float(
                    "m", g_camera.trans_sensitivity, 0.001, 0.03, "move speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset m"):
                g_camera.trans_sensitivity = 0.01

            changed, g_camera.zoom_sensitivity = imgui.slider_float(
                    "z", g_camera.zoom_sensitivity, 0.001, 0.05, "zoom speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset z"):
                g_camera.zoom_sensitivity = 0.01

            changed, g_camera.roll_sensitivity = imgui.slider_float(
                    "ro", g_camera.roll_sensitivity, 0.003, 0.1, "roll speed = %.3f"
                )
            imgui.same_line()
            if imgui.button(label="reset ro"):
                g_camera.roll_sensitivity = 0.03

        # if g_show_help_win:
        #     imgui.begin("Help", True)
        #     imgui.text("Open Gaussian Splatting PLY file \n  by click 'open ply' button")
        #     imgui.text("Use left click & move to rotate camera")
        #     imgui.text("Use right click & move to translate camera")
        #     imgui.text("Press Q/E to roll camera")
        #     imgui.text("Use scroll to zoom in/out")
        #     imgui.text("Use control panel to change setting")
        #     imgui.end()

        imgui.pop_font()

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser(description="NeUVF editor with optional HiDPI support.")
    parser.add_argument("--hidpi", action="store_true", help="Enable HiDPI scaling for the interface.")
    args = parser.parse_args()

    main()

""" Compile different cuda program
raw version:
pip uninstall -y diff-gaussian-rasterization && pip install /raid/yixu/Projects/GaussianSplatting/gaussian-splatting/submodules/diff-gaussian-rasterization

optimized version:
pip uninstall -y diff-gaussian-rasterization && pip install /raid/yixu/Projects/GaussianSplatting/diff-gaussian-rasterization-memory-optimized
"""