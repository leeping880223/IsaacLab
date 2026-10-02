# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""右手 EE 可達範圍取樣的共用函式（給 openarm_workspace_plot.py / openarm_workspace_sim.py 使用）。

做法：在右手 7 個關節的 soft limits 內隨機取樣，直接寫入關節狀態（不跑物理），
讀回 openarm_right_ee_tcp 的位置，得到 EE 可觸及的點雲。

所有座標都轉成「機器人 root 座標系」，也就是 command (object_pose) 使用的座標系。
"""

import math

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

RIGHT_ARM_JOINTS = "openarm_right_joint[1-7]"
EE_BODY = "openarm_right_ee_tcp"
SHOULDER_BODY = "openarm_right_link1"
# 預設姿態下 TCP 的 local +z 朝世界 -z，所以用 local +z 當作夾爪的接近方向
EE_APPROACH_AXIS = (0.0, 0.0, 1.0)
# 檢查會不會撞桌子的手臂 body（手指、TCP 夾方塊時本來就貼近桌面，不檢查）
ARM_LINK_BODIES = "openarm_right_link[1-7]"
HAND_BODY = "openarm_right_hand"


def sample_right_ee_workspace(
    env, num_samples: int, down_angle_deg: float, table=None, link_margin: float = 0.04
) -> dict: #右手可達到哪些點
    """隨機取樣右手關節，回傳 EE 點雲（root 座標系）、「夾爪朝下」遮罩與「不撞桌子」遮罩。

    table 給 (min_xyz, max_xyz)（root 座標系）時，link1~7 的原點落在桌子上方範圍內、且低於
    「桌面 + link_margin [m]」（連桿半徑的粗估），或 hand 原點低於桌面，就判定這組姿勢撞桌子。
    只看 body 原點，是近似判斷。table 為 None 時全部視為不撞。
    """
    scene = env.unwrapped.scene
    robot = scene["robot"]
    num_envs = scene.num_envs
    device = robot.device #取出機器人資料所在的裝置（GPU 或 CPU），存成變數，後面建立新 tensor 時放到同一個裝置上。

    joint_ids, _ = robot.find_joints(RIGHT_ARM_JOINTS) #只取joint id 名稱不用
    ee_id = robot.find_bodies(EE_BODY)[0][0] 
    limits = robot.data.soft_joint_pos_limits.torch[0, joint_ids]  # (7, 2) #只取7個關節&上下限
    lower, upper = limits[:, 0], limits[:, 1]

    root_pos = robot.data.root_pos_w.torch.clone() #clone出來準備修改（機器人位置）
    root_quat = robot.data.root_quat_w.torch.clone() #clone出來準備修改（機器人朝向）
    base_joint_pos = robot.data.default_joint_pos.torch.clone()  #clone出來準備修改（機器人關節位置）
    zero_vel = torch.zeros_like(base_joint_pos)  # 做一個跟 base_joint_pos 一樣大的全 0 tensor，命名為 zero_vel。
    axis = torch.tensor(EE_APPROACH_AXIS, device=device).expand(num_envs, 3) #把目標pose複製到每個環境中
    world_down = torch.tensor([0.0, 0.0, -1.0], device=device) #世界的座標軸,把每個環境的夾爪方向跟它做內積，就能知道夾爪偏離正下方多少度。
    cos_thresh = math.cos(math.radians(down_angle_deg)) #把自行設定允許角度最大值轉換為cos值

    link_ids = robot.find_bodies(ARM_LINK_BODIES)[0] + robot.find_bodies(HAND_BODY)[0]
    # 每個 body 的高度餘量：手臂連桿用 link_margin，hand 用 0
    link_margins = torch.tensor([link_margin] * (len(link_ids) - 1) + [0.0], device=device)
    if table is not None:
        t_lo = torch.tensor(table[0], device=device)
        t_hi = torch.tensor(table[1], device=device)

    points, down_mask, free_mask = [], [], []
    num_batches = math.ceil(num_samples / num_envs) # 要跑幾輪 = 透過需要的總採樣數量/ 環境數量 可以得到要跑幾輪？ 除不盡時要無條件進位。
    for _ in range(num_batches):
        q = lower + (upper - lower) * torch.rand(num_envs, len(joint_ids), device=device) #角度抓隨機變數 （下限角度 ＋（可活動的範圍）x (0~1的隨機數字)
        joint_pos = base_joint_pos.clone()
        joint_pos[:, joint_ids] = q #更改關節角度
        robot.write_joint_state_to_sim_index(position=joint_pos, velocity=zero_vel) #把關節角度與速度寫進issac-sim內（速度＝0）

        # 讀 body 資料時會重算 FK
        ee_pos_w = robot.data.body_pos_w.torch[:, ee_id] #把末端的位置讀出來（世界座標系）
        ee_quat_w = robot.data.body_quat_w.torch[:, ee_id] #把末端的角度讀出來（世界座標系）
        points.append(quat_apply_inverse(root_quat, ee_pos_w - root_pos))#把末端跟root的座標差算出來（世界座標系）,再把它轉到機器人座標系,並存進points 這個list當中
        approach_w = quat_apply(ee_quat_w, axis) #把axis從ee座標系轉換為world
        if table is None:
            free_mask.append(torch.ones(num_envs, dtype=torch.bool, device=device))
        else:
            # 手臂 body 轉到 root 座標系 (num_envs, n_links, 3)
            link_w = robot.data.body_pos_w.torch[:, link_ids] - root_pos.unsqueeze(1)
            n = len(link_ids)
            link_r = quat_apply_inverse(root_quat.repeat_interleave(n, 0), link_w.reshape(-1, 3)).reshape(-1, n, 3)
            m = link_margins.view(1, -1)
            over_table = ((link_r[..., 0] > t_lo[0] - m) & (link_r[..., 0] < t_hi[0] + m)
                          & (link_r[..., 1] > t_lo[1] - m) & (link_r[..., 1] < t_hi[1] + m))
            hit = over_table & (link_r[..., 2] < t_hi[2] + m)
            free_mask.append(~hit.any(dim=1))
        down_mask.append((approach_w @ world_down) > cos_thresh) #把ee的接近方向與重力方向內積,可以得到夾角的cos,再與自行設定的允許角度最大值轉換為cos值做比較,小於允許最大夾角的在list中顯示true,大於顯示False,並且都會存(cos0~180 角度越大數值越小)

    points = torch.cat(points)[:num_samples] #位置串接並截斷#cat 是 concatenate（串接）預設是上下接,把很多輪的（ num_envs,3）串成 （輪數* num_envs,3),再截斷（[:n]）保留num_samples個
    down_mask = torch.cat(down_mask)[:num_samples] #否朝下的 True/False串接並截斷
    free_mask = torch.cat(free_mask)[:num_samples]

    # 右肩位置（預設姿態），方便看距離
    robot.write_joint_state_to_sim_index(position=base_joint_pos, velocity=zero_vel) #回到初始位置
    shoulder_w = robot.data.body_pos_w.torch[0, robot.find_bodies(SHOULDER_BODY)[0][0]] #看肩膀的位置（世界座標系）
    shoulder = quat_apply_inverse(root_quat[:1], (shoulder_w - root_pos[0]).unsqueeze(0))[0] #把肩膀朝向轉為機器人座標系

    return {"points": points, "down_mask": down_mask, "free_mask": free_mask, "shoulder": shoulder}


def get_task_boxes(env_cfg) -> dict:#任務要求手去哪裡
    """從 env cfg 讀出方塊生成範圍與目標範圍，回傳 {name: (min_xyz, max_xyz)}（root 座標系）。"""
    robot_pos = env_cfg.scene.robot.init_state.pos #機器人初始位置
    robot_rot = env_cfg.scene.robot.init_state.rot #機器人初始姿態
    if tuple(robot_rot) != (0.0, 0.0, 0.0, 1.0):
        print(f"[WARN] robot init rot = {robot_rot}，不是單位四元數；方塊範圍只扣掉位置平移，可能不準。")#看機器人有沒有旋轉

    obj_pos = env_cfg.scene.object.init_state.pos #方塊初始位置
    pose_range = env_cfg.events.reset_object_position.params["pose_range"]#方塊隨機位置
    cube_min, cube_max = [], [] 
    for i, key in enumerate("xyz"):#把方塊可以動範圍的極端點轉為座標
        lo, hi = pose_range.get(key, (0.0, 0.0))
        cube_min.append(obj_pos[i] + lo - robot_pos[i])
        cube_max.append(obj_pos[i] + hi - robot_pos[i])

    r = env_cfg.commands.object_pose.ranges #移動的目標範圍
    target_min = [r.pos_x[0], r.pos_y[0], r.pos_z[0]]
    target_max = [r.pos_x[1], r.pos_y[1], r.pos_z[1]]

    return {"cube": (cube_min, cube_max), "target": (target_min, target_max)}


def get_table_box(env) -> tuple[list, list] | None:#桌子在哪裡
    """讀 env 0 桌子「碰撞體」的包圍盒（含桌腳；視覺模型有額外的高物件，不用），回傳 (min_xyz, max_xyz)（root 座標系）；找不到回傳 None。"""
    from pxr import Usd, UsdGeom

    import isaaclab.sim as sim_utils

    scene = env.unwrapped.scene
    prim = sim_utils.get_current_stage().GetPrimAtPath(f"{scene.env_prim_paths[0]}/Table/Collisions")
    if not prim.IsValid():
        print("[WARN] 找不到桌子 prim，圖上不畫桌子。")
        return None
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    # 跟 get_task_boxes 一樣只扣 root 平移（假設機器人沒有旋轉）
    root_pos = scene["robot"].data.root_pos_w.torch[0].tolist()
    lo = [v - r for v, r in zip(bounds.GetMin(), root_pos)]
    hi = [v - r for v, r in zip(bounds.GetMax(), root_pos)]
    return lo, hi


def box_grid(box_min, box_max, steps: int, device) -> torch.Tensor:#範圍切割成點格,用來測試手臂到不到得了
    """在範圍框內產生均勻格點（某軸範圍為 0 時只取一層）。"""
    axes = []
    for lo, hi in zip(box_min, box_max):
    # zip 把每一軸的最小、最大配成一組：[(minX,maxX), (minY,maxY), (minZ,maxZ)]
    # 迴圈跑 3 次，每次拆成 lo（最小）、hi（最大）
        n = 1 if abs(hi - lo) < 1e-6 else steps #透過step 看要切多細的格子
        axes.append(torch.linspace(lo, hi, n, device=device)) #把每一格的xyz座標填入
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
    # 排出三軸刻度的所有組合（meshgrid 產生 X/Y/Z 三張表 → stack 合成每格 [x,y,z] → reshape 攤平），回傳格點 (點數, 3)

def check_coverage(grid: torch.Tensor, cloud: torch.Tensor, tol: float) -> torch.Tensor:#逐一檢查格點附近有無點雲
    """格點附近 tol 內有點雲 → 視為可達。回傳 bool (G,)。"""
    #點雲cloud = 手到得了的地方，格點grid = 任務要手去的地方。tol 容許距離預設是3cm
    if cloud.numel() == 0:# 點雲是空的 → 所有格點都回傳False
        return torch.zeros(len(grid), dtype=torch.bool, device=grid.device)
    reachable = []
    for chunk in cloud.split(65536):#點雲分批計算以免gpu爆掉
        reachable.append(torch.cdist(grid, chunk).min(dim=1).values)#計算grid跟分割完的cloud(chunk)計算距離（torch.cdist()),在裡面沿橫向（dim=1）找最小值
    return torch.stack(reachable, dim=1).min(dim=1).values < tol # 每個格點的最近距離 < tol → True


def print_report(
    boxes: dict, grids: dict, reach: dict, shoulder: torch.Tensor, tol: float, down_angle: float, table=None,
    frame: str = "root",
):#印出百分比結果
    frame_name = {"root": "root 座標系", "env": "環境原點座標系"}[frame]
    print(f"\n========== 右手可達性報告（{frame_name}, 公尺） ==========")
    print(f"右肩 (openarm_right_link1): {[round(v, 3) for v in shoulder.tolist()]}")
    if table is not None:
        print(f"桌子 min={[round(v, 3) for v in table[0]]} max={[round(v, 3) for v in table[1]]}"
              f"（桌面 z={table[1][2]:.3f}，肩膀高出桌面 {shoulder[2].item() - table[1][2]:.3f}）")
    print(f"判定：格點 {tol * 100:.1f} cm 內有「夾爪朝下（<{down_angle:.0f}°）」的 EE 取樣點 → 可達")
    for name in boxes:
        lo, hi = boxes[name]
        ok = reach[name]
        print(f"\n[{name}] 範圍 min={[round(v, 3) for v in lo]} max={[round(v, 3) for v in hi]}")
        print(f"  可達格點: {int(ok.sum())}/{len(ok)} ({100.0 * ok.float().mean():.1f}%)")
        bad = grids[name][~ok]
        if len(bad) > 0:
            print(f"  不可達格點範圍: min={[round(v, 3) for v in bad.min(0).values.tolist()]}"
                  f" max={[round(v, 3) for v in bad.max(0).values.tolist()]}")
    print("==========================================================\n")
