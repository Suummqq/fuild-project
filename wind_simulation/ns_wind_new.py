import argparse

import numpy as np

import taichi as ti


# How to run:
#   `python stable_fluid.py`: use the jacobi iteration to solve the linear system.
#   `python stable_fluid.py -S`: use a sparse matrix to do so.
parser = argparse.ArgumentParser()
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
dt = 0.3       #间隔步长
p_jacobi_iters = 50  # 40 for a quicker but less accurate result
f_strength = 10000.0        #外部施加的力的强度
curl_strength = 0           #流体中添加涡旋的强度
time_c = 2                  #时间缩放系数
maxfps = 60                 #模拟的最大帧率
dye_decay = 1 - 1 / (maxfps * time_c)       #颜料的衰减系数
force_radius = res / 2.0            #施加力的影响半径
debug = True
radius = 30
center = [0.5,0.5]

arch = args.arch
if arch in ["x64", "cpu", "arm64"]:
    ti.init(arch=ti.cpu,cpu_max_num_threads=1)
elif arch in ["cuda", "gpu"]:
    ti.init(arch=ti.cuda,debug=debug)
else:
    raise ValueError("Only CPU and CUDA backends are supported for now.")

_velocities = ti.Vector.field(2, float, shape=(res, res))       #二维向量场，表示流体的速度场
_new_velocities = ti.Vector.field(2, float, shape=(res, res))   #二维向量场，用于存储更新后的速度场。
velocity_divs = ti.field(float, shape=(res, res))               #二维浮点数场，用于存储速度场的散度
velocity_curls = ti.field(float, shape=(res, res))              #二维浮点数场，用于存储速度场的旋度
_pressures = ti.field(float, shape=(res, res))                  #二维浮点数场，表示流体的压力场
_new_pressures = ti.field(float, shape=(res, res))              #二维浮点数场，用于存储更新后的压力场
_dye_buffer = ti.Vector.field(3, float, shape=(res, res))       #三维向量场，用于存储颜色或染料的信息
_new_dye_buffer = ti.Vector.field(3, float, shape=(res, res))   #三维向量场，用于存储更新后的颜色或染料信息
solid_fild = ti.field(float, shape=(res, res))               #障碍物


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

#坐标轴范围限制在0-res之间，并返回该点的速度场
@ti.func
def sample(qf, u, v):
    I = ti.Vector([int(u), int(v)])
    I = ti.max(0, ti.min(res - 1, I))
    return qf[I]*(1-solid_fild[I])
def sample_v(qf, u, v):
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
    b = sample(vf, iu - 1, iv)      #左侧速度场
    c = sample(vf, iu, iv + 1)      #上侧侧速度场
    d = sample(vf, iu - 1, iv + 1)  #左上侧速度场
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
    # ti.static_print(vf)
    for i, j in vf:
        # print(i,j)

        p = ti.Vector([i, j]) + 0.5
        # print(p)
        p = backtrace(vf, p, dt)        #回溯位置后的p点
        new_qf[i, j] = bilerp(qf, p)        #回溯后该点p的速度赋值给当前点

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
        if 0<i< res-1 and 0<j < res-1:
            if (solid_fild[i,j]==0 and solid_fild[i+1,j]!=0):        #碰到墙壁给个反方向的速度
                vr.x = -vc.x        #x是第一个向量，y是第二个向量
            if (solid_fild[i,j]==0 and solid_fild[i,j+1]!=0):
                vb.y = -vc.y
            if (solid_fild[i,j]==0 and solid_fild[i,j-1]!=0):
                vt.y = -vc.y
        # if i == res - 1:
        #     vr.x = vc.x
        # # if j == 0:
        # #     vb.y = -vc.y
        # if j == res - 1:
        #     vt.y = vc.y
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
        new_pf[i, j] = (pl + pr + pb + pt - div) * 0.25     #根据Pressure solve的公式推导p_{ij}


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = sample(pf, i - 1, j)
        pr = sample(pf, i + 1, j)
        pb = sample(pf, i, j - 1)
        pt = sample(pf, i, j + 1)
        vf[i, j] -= dt * ti.Vector([pr - pl, pt - pb])


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



#jacobi迭代法求解速度
def solve_pressure_jacobi():
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

# @ti.kernel
# def set_boundary_velocity(vf: ti.template(), sf: ti.template()):
#     for i,j in vf:
#         if j>=int(9*res/20) and j<=int(11*res/20) and ((i + 0.5) - res*center[0])**2 + ((j + 0.5) - res*center[1]) ** 2 > radius ** 2:
#             vf[i, j][0] = 2.0  # 左边界为入口，设置速度为1
#             vf[i, j][1] = 0.0
#         elif ((i + 0.5) - res*center[0])**2 + ((j + 0.5) - res*center[1]) ** 2 <= radius ** 2:
#             sf[i,j] = 1         #设置固体区域为1
@ti.kernel
def set_boundary_velocity(vf: ti.template(), sf: ti.template()):
    for i,j in vf:
        if ((i + 0.5) - res * center[0]) ** 2 + ((j + 0.5) - res * center[1]) ** 2 <= radius ** 2:
            sf[i,j] = 1
        if i== 0 :
            vf[i, j][0] = 100.0  # 左边界为入口，设置速度为10

@ti.kernel
def extrapolate(vf:ti.template()):
    for i,j in vf:
        if i == res-1:
            vf[i,j][0] = vf[i-1,j][0]
            vf[i, j][1] = vf[i - 1, j][1]




def setup():

    advect(velocities_pair.cur,velocities_pair.cur, velocities_pair.nxt)
    # advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)                   #上色
    # print( velocities_pair.nxt[0,260])
    velocities_pair.swap()
    divergence(velocities_pair.cur)
    solve_pressure_jacobi()
    subtract_gradient(velocities_pair.cur, pressures_pair.cur)
    # extrapolate(velocities_pair.cur)




def main():
    gui = ti.GUI("风场模拟", (res, res))
    while gui.running:
        set_boundary_velocity(velocities_pair.cur,solid_fild)
        setup()
        gui.set_image(velocities_pair.cur.to_numpy() * 0.02 + 0.5)
        gui.circle(pos=center, radius=radius, color=0xFFFFFF)
        gui.show()

if __name__ == '__main__':
    main()