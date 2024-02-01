# References:
# http://developer.download.nvidia.com/books/HTML/gpugems/gpugems_ch38.html
# https://github.com/PavelDoGreat/WebGL-Fluid-Simulation
# https://www.bilibili.com/video/BV1ZK411H7Hc?p=4
# https://github.com/ShaneFX/GAMES201/tree/master/HW01

import argparse

import numpy as np

import taichi as ti

# How to run:
#   `python stable_fluid.py`: use the jacobi iteration to solve the linear system.
#   `python stable_fluid.py -S`: use a sparse matrix to do so.
parser = argparse.ArgumentParser()
parser.add_argument(
    "-S",
    "--use-sp-mat",
    action="store_true",
    help="Solve Poisson's equation by using a sparse matrix",
)
parser.add_argument(
    "-a",
    "--arch",
    required=False,
    default="cuda",
    dest="arch",
    type=str,
    help="The arch (backend) to run this example on",
)
args, unknowns = parser.parse_known_args()

res = 512
dt = 0.03       #间隔步长
p_jacobi_iters = 500  # 40 for a quicker but less accurate result
f_strength = 10000.0        #外部施加的力的强度
curl_strength = 0           #流体中添加涡旋的强度
time_c = 2                  #时间缩放系数
maxfps = 60                 #模拟的最大帧率
dye_decay = 1 - 1 / (maxfps * time_c)       #颜料的衰减系数
force_radius = res / 2.0            #施加力的影响半径
debug = False

use_sparse_matrix = args.use_sp_mat
arch = args.arch
if arch in ["x64", "cpu", "arm64"]:
    ti.init(arch=ti.cpu,cpu_max_num_threads=1)
elif arch in ["cuda", "gpu"]:
    ti.init(arch=ti.cuda,debug=debug)
else:
    raise ValueError("Only CPU and CUDA backends are supported for now.")

if use_sparse_matrix:
    print("Using sparse matrix")
else:
    print("Using jacobi iteration")

_velocities = ti.Vector.field(2, float, shape=(res, res))       #二维向量场，表示流体的速度场
_new_velocities = ti.Vector.field(2, float, shape=(res, res))   #二维向量场，用于存储更新后的速度场。
velocity_divs = ti.field(float, shape=(res, res))               #二维浮点数场，用于存储速度场的散度
velocity_curls = ti.field(float, shape=(res, res))              #二维浮点数场，用于存储速度场的旋度
_pressures = ti.field(float, shape=(res, res))                  #二维浮点数场，表示流体的压力场
_new_pressures = ti.field(float, shape=(res, res))              #二维浮点数场，用于存储更新后的压力场
_dye_buffer = ti.Vector.field(3, float, shape=(res, res))       #三维向量场，用于存储颜色或染料的信息
_new_dye_buffer = ti.Vector.field(3, float, shape=(res, res))   #三维向量场，用于存储更新后的颜色或染料信息

#新旧场的交换
class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)

if use_sparse_matrix:
    # use a sparse matrix to solve Poisson's pressure equation.
    @ti.kernel
    def fill_laplacian_matrix(A: ti.types.sparse_matrix_builder()):
        for i, j in ti.ndrange(res, res):
            row = i * res + j
            center = 0.0
            if j != 0:
                A[row, row - 1] += -1.0
                center += 1.0
            if j != res - 1:
                A[row, row + 1] += -1.0
                center += 1.0
            if i != 0:
                A[row, row - res] += -1.0
                center += 1.0
            if i != res - 1:
                A[row, row + res] += -1.0
                center += 1.0
            A[row, row] += center

    N = res * res
    K = ti.linalg.SparseMatrixBuilder(N, N, max_num_triplets=N * 6)
    F_b = ti.ndarray(ti.f32, shape=N)

    fill_laplacian_matrix(K)
    L = K.build()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(L)
    solver.factorize(L)

#坐标轴范围限制在0-res之间，并返回该点的速度场
@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = ti.max(0, ti.min(res - 1, I))
    return qf[I]


@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

#双线性插值
@ti.func
def bilerp(vf, p):      #vf是向量场 p是xy坐标系中点
    u, v = p
    s, t = u - 0.5, v - 0.5
    # floor
    iu, iv = ti.floor(s), ti.floor(t)       #取下界
    # fract
    fu, fv = s - iu, t - iv
    # print(fu,fv)
    a = sample(vf, iu, iv)          #当前点的速度场
    b = sample(vf, iu + 1, iv)      #右侧速度场
    c = sample(vf, iu, iv + 1)      #下侧速度场
    d = sample(vf, iu + 1, iv + 1)  #右下侧速度场
    return lerp(lerp(a, b, fu), lerp(c, d, fu), fv)     #多次插值获取当前点的速度场


# 3rd order Runge-Kutta 回溯位置 类似半拉格朗日方法获取上一个时间步的对流速度并更新到现在
@ti.func
def backtrace(vf: ti.template(), p, dt_: ti.template()):
    v1 = bilerp(vf, p)          #p点的速度场
    p1 = p - 0.5 * dt_ * v1
    v2 = bilerp(vf, p1)
    p2 = p - 0.75 * dt_ * v2
    v3 = bilerp(vf, p2)
    p -= dt_ * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
    return p


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    ti.static_print(vf)
    for i, j in vf:
        # print(i,j)
        p = ti.Vector([i, j]) + 0.5
        # print(p)
        p = backtrace(vf, p, dt)        #回溯位置后的p点
        new_qf[i, j] = bilerp(qf, p) * dye_decay        #回溯后该点p的速度赋值给当前点


@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(), imp_data: ti.types.ndarray()):
    g_dir = -ti.Vector([0, 9.8]) * 300
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]             #鼠标点击的坐标位置
        mdir = ti.Vector([imp_data[0], imp_data[1]])    #鼠标移动的方向
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)       #鼠标与坐标的距离
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 / force_radius)             #高斯衰减力进行模拟

        dc = dyef[i, j]
        a = dc.norm()

        momentum = (mdir * f_strength * factor + g_dir * a / (1 + a)) * dt

        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * (4 / (res / 15) ** 2)) * ti.Vector([imp_data[4], imp_data[5], imp_data[6]])

        dyef[i, j] = dc

#整个场的散度 （散度要为0）
@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        vc = sample(vf, i, j)
        # print(vc[0])
        if i == 0:                      #碰到墙壁给个反方向的速度
            vl.x = -vc.x        #x是第一个向量，y是第二个向量
        if i == res - 1:
            vr.x = -vc.x
        if j == 0:
            vb.y = -vc.y
        if j == res - 1:
            vt.y = -vc.y
        velocity_divs[i, j] = (vr.x - vl.x + vt.y - vb.y) * 0.5


@ti.kernel
def vorticity(vf: ti.template()):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        velocity_curls[i, j] = (vr.y - vl.y - vt.x + vb.x) * 0.5


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= 0.5 * ti.Vector([pr - pl, pt - pb])


@ti.kernel
def enhance_vorticity(vf: ti.template(), cf: ti.template()):
    # anti-physics visual enhancement...
    for i, j in vf:
        cl = sample(cf, i - 1, j)
        cr = sample(cf, i + 1, j)
        cb = sample(cf, i, j - 1)
        ct = sample(cf, i, j + 1)
        cc = sample(cf, i, j)
        force = ti.Vector([abs(ct) - abs(cb), abs(cl) - abs(cr)]).normalized(1e-3)
        force *= curl_strength * cc
        vf[i, j] = ti.min(ti.max(vf[i, j] + force * dt, -1e3), 1e3)


@ti.kernel
def copy_divergence(div_in: ti.template(), div_out: ti.types.ndarray()):
    for I in ti.grouped(div_in):
        div_out[I[0] * res + I[1]] = -div_in[I]


@ti.kernel
def apply_pressure(p_in: ti.types.ndarray(), p_out: ti.template()):
    for I in ti.grouped(p_out):
        p_out[I] = p_in[I[0] * res + I[1]]


def solve_pressure_sp_mat():
    copy_divergence(velocity_divs, F_b)
    x = solver.solve(F_b)
    apply_pressure(x, pressures_pair.cur)

#jacobi迭代法求解速度
def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()


def step(mouse_data):
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)       #对流
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)                   #上色
    velocities_pair.swap()                                                      #更新当前速度信息
    dyes_pair.swap()                                                            #更新当前颜色信息

    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)               #根据外力更新速度
    #求散度
    divergence(velocities_pair.cur)
    #计算涡度
    if curl_strength:
        vorticity(velocities_pair.cur)
        enhance_vorticity(velocities_pair.cur, velocity_curls)

    if use_sparse_matrix:
        solve_pressure_sp_mat()
    else:
        solve_pressure_jacobi()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)

    if debug:
        divergence(velocities_pair.cur)
        div_s = np.sum(velocity_divs.to_numpy())
        print(f"divergence={div_s}")


class MouseDataGen:
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction 归一化后的鼠标方向
        # [2:4]: current mouse xy   当前鼠标位置
        # [4:7]: color  颜色
        mouse_data = np.zeros(8, dtype=np.float32)
        if gui.is_pressed(ti.GUI.LMB):
            mxy = np.array(gui.get_cursor_pos(), dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                # Set lower bound to 0.3 to prevent too dark colors
                self.prev_color = (np.random.rand(3) * 0.7) + 0.3
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                mouse_data[0], mouse_data[1] = mdir[0], mdir[1]
                mouse_data[2], mouse_data[3] = mxy[0], mxy[1]
                mouse_data[4:7] = self.prev_color
                self.prev_mouse = mxy
        else:
            self.prev_mouse = None
            self.prev_color = None
        return mouse_data


def reset():
    velocities_pair.cur.fill(0)
    pressures_pair.cur.fill(0)
    dyes_pair.cur.fill(0)


def main():
    global debug, curl_strength
    visualize_d = True  # visualize dye (default)
    visualize_v = False  # visualize velocity
    visualize_c = False  # visualize curl

    paused = False

    gui = ti.GUI("Stable Fluid", (res, res))
    md_gen = MouseDataGen()

    while gui.running:
        if gui.get_event(ti.GUI.PRESS):
            e = gui.event
            if e.key == ti.GUI.ESCAPE:
                break
            elif e.key == "r":
                paused = False
                reset()
            elif e.key == "s":
                if curl_strength:
                    curl_strength = 0
                else:
                    curl_strength = 7
            elif e.key == "v":
                visualize_v = True
                visualize_c = False
                visualize_d = False
            elif e.key == "d":
                visualize_d = True
                visualize_v = False
                visualize_c = False
            elif e.key == "c":
                visualize_c = True
                visualize_d = False
                visualize_v = False
            elif e.key == "p":
                paused = not paused
            elif e.key == "d":
                debug = not debug

        # Debug divergence:
        # print(max((abs(velocity_divs.to_numpy().reshape(-1)))))

        if not paused:
            mouse_data = md_gen(gui)
            step(mouse_data)
        if visualize_c:
            vorticity(velocities_pair.cur)
            gui.set_image(velocity_curls.to_numpy() * 0.03 + 0.5)
        elif visualize_d:
            gui.set_image(dyes_pair.cur)
        elif visualize_v:
            gui.set_image(velocities_pair.cur.to_numpy() * 0.01 + 0.5)
        gui.show()


if __name__ == "__main__":
    main()