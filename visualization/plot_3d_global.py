import torch 
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio

# 여러개의 모션을 하나의 플롯에다가 그리는 방법

def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4, footer_text=None, footer_fontsize=12):
    matplotlib.use('Agg')
    
    
    joints, out_name, title = args
    
    data = joints.copy().reshape(len(joints), -1, 3)
    
    nb_joints = joints.shape[1]
    smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):

        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)
        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)
        fig = plt.figure(figsize=(480/96., 320/96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(10, 10), dpi=96)
        if title is not None :
            wraped_title = '\n'.join(wrap(title, 50))
            fig.suptitle(wraped_title, fontsize=20)
        ax = p3.Axes3D(fig)
        
        init()
        
        ax.lines = []
        ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 10.5

        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        
        if footer_text:
            # 여백 확보: 하단 마진을 약간 늘려 둡니다
            fig.subplots_adjust(bottom=0.13)
            # (x, y) = (0.5, 0.015) 지점에 가운데 정렬
            fig.text(
                0.5, 0.03, footer_text,
                ha='center', va='bottom',
                fontsize=footer_fontsize
            )

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')


        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):

            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)


        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
        if out_name is not None : 
            plt.savefig(out_name, dpi=96)
            plt.close()
            
        else : 
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)

            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                                newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number) : 
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


# def plot_3d_motion_subplot(joints_list, title_list=None, figsize=(15, 5)):
#     num_plots = len(joints_list)
#     fig = plt.figure(figsize=(figsize[0]*num_plots, figsize[1]), dpi=150)

#     def draw_single_motion(ax, joints, title):
#         data = joints.copy().reshape(len(joints), -1, 3)
#         nb_joints = joints.shape[1]
#         smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
#         limits = 1000 if nb_joints == 21 else 2
#         MINS = data.min(axis=0).min(axis=0)
#         MAXS = data.max(axis=0).max(axis=0)
#         colors = ['red', 'blue', 'black', 'red', 'blue',
#                   'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
#                   'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
        
#         height_offset = MINS[1]
#         data[:, :, 1] -= height_offset
#         trajec = data[:, 0, [0, 2]]

#         data[..., 0] -= data[:, 0:1, 0]
#         data[..., 2] -= data[:, 0:1, 2]

#         ax.set_xlim(-limits, limits)
#         ax.set_ylim(-limits, limits)
#         ax.set_zlim(0, limits)
#         ax.grid(b=False)
#         ax.view_init(elev=110, azim=-90)
#         ax.dist = 10.5
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.axis('off')

#         def plot_xzPlane():
#             verts = [
#                 [MINS[0], 0, MINS[2]],
#                 [MINS[0], 0, MAXS[2]],
#                 [MAXS[0], 0, MAXS[2]],
#                 [MAXS[0], 0, MINS[2]]
#             ]
#             xz_plane = Poly3DCollection([verts])
#             xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
#             ax.add_collection3d(xz_plane)

#         plot_xzPlane()

#         for chain, color in zip(smpl_kinetic_chain, colors):
#             ax.plot3D(data[-1, chain, 0], data[-1, chain, 1], data[-1, chain, 2], linewidth=4.0, color=color)

#         if title is not None:
#             wrapped_title = '\n'.join(wrap(title, 50))
#             ax.set_title(wrapped_title, fontsize=15)

#     for idx, joints in enumerate(joints_list):
#         ax = fig.add_subplot(1, num_plots, idx + 1, projection='3d')
#         title = title_list[idx] if title_list is not None else None
#         draw_single_motion(ax, joints, title)

#     plt.tight_layout()

#     io_buf = io.BytesIO()
#     fig.savefig(io_buf, format='raw', dpi=150)
#     io_buf.seek(0)

#     arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
#                      newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
#     io_buf.close()
#     plt.close()
    
#     return torch.from_numpy(arr)

# import torch 
# import matplotlib.pyplot as plt
# import numpy as np
# import io
# import matplotlib
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from textwrap import wrap
# import imageio

# matplotlib.use('Agg')

# def plot_3d_motion_subplot(joints_list, title_list=None, figsize=(20, 10), dpi=200):
#     num_plots = len(joints_list)
#     fig = plt.figure(figsize=(figsize[0]*num_plots, figsize[1]), dpi=dpi)

#     def draw_single_motion(ax, joints, title):
#         data = joints.copy().reshape(len(joints), -1, 3)
#         nb_joints = joints.shape[1]
#         smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
        
#         MINS = data.min(axis=(0,1))
#         MAXS = data.max(axis=(0,1))
#         max_range = (MAXS - MINS).max() * 0.5
#         mid_x = (MAXS[0] + MINS[0]) / 2
#         mid_y = (MAXS[1] + MINS[1]) / 2
#         mid_z = (MAXS[2] + MINS[2]) / 2

#         ax.set_xlim(mid_x - max_range, mid_x + max_range)
#         ax.set_ylim(mid_y - max_range, mid_y + max_range)
#         ax.set_zlim(mid_z - max_range, mid_z + max_range)
#         ax.grid(False)
#         ax.view_init(elev=110, azim=-90)
#         ax.dist = 10
#         ax.axis('off')

#         for chain, color in zip(smpl_kinetic_chain, ['red', 'blue', 'black', 'red', 'blue']):
#             ax.plot3D(data[-1, chain, 0], data[-1, chain, 1], data[-1, chain, 2], linewidth=3, color=color)

#         if title:
#             wrapped_title = '\n'.join(wrap(title, 50))
#             ax.set_title(wrapped_title, fontsize=15)

#     for idx, joints in enumerate(joints_list):
#         ax = fig.add_subplot(1, num_plots, idx + 1, projection='3d')
#         title = title_list[idx] if title_list else None
#         draw_single_motion(ax, joints, title)

#     plt.tight_layout()

#     io_buf = io.BytesIO()
#     fig.savefig(io_buf, format='png', dpi=dpi)
#     io_buf.seek(0)

#     arr = imageio.imread(io_buf)
#     io_buf.close()
#     plt.close()

#     return torch.from_numpy(arr)




def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None, footer_text=None, footer_fontsize=None) : 
    
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size) : 
        out.append(plot_3d_motion([smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None], footer_text=footer_text, footer_fontsize=footer_fontsize))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=20)
    out = torch.stack(out, axis=0)
    return out
    
