# save as generate_racket_fixed.py
import numpy as np
from stl import mesh

def box_from_center(center, size):
    """
    center: (cx,cy,cz)
    size: (sx,sy,sz) full lengths
    returns: numpy-stl Mesh for a box (12 triangles)
    """
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    # 8 verts
    v = np.array([
        [cx - sx, cy - sy, cz - sz],
        [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz],
        [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz],
        [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz],
        [cx - sx, cy + sy, cz + sz],
    ])
    faces = np.array([
        [0,3,1], [1,3,2],  # bottom (-z)
        [4,5,7], [5,6,7],  # top (+z)
        [0,1,4], [1,5,4],  # -y face
        [2,3,6], [3,7,6],  # +y face
        [1,2,5], [2,6,5],  # +x face
        [0,4,3], [3,4,7],  # -x face
    ])
    m = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            m.vectors[i][j] = v[f[j], :]
    return m

if __name__ == "__main__":
    # 参数（米），可按需修改
    handle_length = 0.40     # 手柄长度（从底端到拍颈连接处）
    handle_w = 0.03         # 手柄横向宽度
    handle_th = 0.03        # 手柄厚度

    head_w = 0.03            # 拍面宽（x 方向半宽合计）
    head_h = 0.3            # 拍面高（y 方向）
    head_th = 0.40           # 拍面厚度（z 方向）

    # 放置方式：
    # - 手柄从 z = -handle_length 到 z = 0（上端在 z=0）
    # - 手柄中心在 z = -handle_length/2
    handle_center = (0.0, 0.0, -handle_length / 2.0)
    handle_size = (handle_w, handle_th, handle_length)  # 注意 box_from_center size 是 (sx,sy,sz)

    # 拍面紧接在手柄上端放置，厚度沿 +z
    # 拍面中心 z = head_th/2 （因为拍面占 z∈[0, head_th]）
    head_center = (0.0, 0.0, head_th / 2.0)
    head_size = (head_w, head_h, head_th)

    handle_mesh = box_from_center(handle_center, handle_size)
    head_mesh = box_from_center(head_center, head_size)

    # 合并两个 mesh
    combined = mesh.Mesh(np.concatenate([handle_mesh.data, head_mesh.data]))

    out_name = "racket_fixed_orient.stl"
    combined.save(out_name)
    print(f"Saved {out_name}")
