# OpenArm 兩份 USD 的夾持相關差異

比較日期：2026-10-19。以下是在 `isaac-rl2` Docker 內用 Isaac Sim 6.1 / OpenUSD 做的唯讀檢查。今天 Isaac Sim 6.0 視窗中看到的官方資產不一定與這份 6.1 檔案完全相同；若要比較視窗中的 6.0 資產，需要另做一次檢查。

## 先看結論

- 配置版把右手兩指各往外移約 **1 mm**，靜態零位間隙因此增加約 **2 mm**；關節行程、drive、摩擦材質和部分接觸設定也不同。
- 兩份 USD 啟用的指片碰撞網格在各自手指座標下幾乎相同。**配置版仍保留右指負 Y 縮放**，不能稱為「已修復負縮放的 USD」。
- 昨天的 10 mm 方塊接觸初測中，配置版原始檔的**兩指都有非零接觸力**；把右指負縮放烘入頂點後，右指也有接觸。因此這個測試沒有重現「原版右指全程 0 N」的說法。
- 這是 USD 屬性與一組簡化接觸實驗的比較，**不能單靠它判定哪份 USD 的夾力較好**。要比較夾力，還要固定物件、姿勢、控制、物理材質及時間步長再做 A/B。

## 比較對象

- 舊／官方：`OPENARM_BI_CFG.spawn.usd_path`，解析為 `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Robots/OpenArm/openarm_bimanual/openarm_bimanual.usd`。設定來源：[openarm.py](../../source/isaaclab_assets/isaaclab_assets/robots/openarm.py)。
- 新／配置包：`/home/Intern_1/openarm_sim_pack/openarm_configured.usd`。專案內的[配置版 USD](../../source/isaaclab_assets/data/Robots/OpenArm/openarm_configured.usd)與它的 SHA-256 都是 `a6bbc93d945609d7840b730e524113ed8630ebd519af2f848bec51512c34ba14`。

以下「舊／新」專指上面兩份檔。配置包 README 提到的「舊 STL collider」是**配置包的早期版本**，不是這份 IsaacLab 原版 USD。

**鏡像碰撞補查（2026-10-19）：**兩份 USD 各有 18 個 `xformOp:scale.y < 0`。舊檔右手右指碰撞網格的負縮放位於 `/openarm/openarm_right_right_finger/collisions/openarm_right_right_finger_collision`；配置包啟用的右手右指碰撞網格位於 `/World/openarm/follower_r_right_finger/visuals/follower_r_right_finger_visual/Scene/mesh`，其上層 `.../follower_r_right_finger_visual` 的 `xformOp:scale.y=-0.001`。因此配置包不是這個負縮放問題的修復後對照；以下網格位置僅表示 USD 靜態幾何，不能推斷 PhysX 烹煮後碰撞殼的位置。可用 `/home/Intern_1/openarm_mirror_audit.py` 重查屬性。

**接觸初測（2026-10-19）：**用 `/home/Intern_1/openarm_finger_contact_ab.py` 在配置包同一份 USD 上做 240 步、480 Hz、零重力、固定 10 mm 方塊、右夾爪由 7 mm 開到 0 mm 的 A/B。`original` 保留右指負 Y 縮放；`fixed` 只把這根指片的 Y 反射烘入網格點並將縮放改正值（記憶體修改，原檔不變）。方塊過濾的法向接觸力峰值：原版左指 `113.42 N`、右指 `95.43 N`；修正版左指 `114.82 N`、右指 `100.44 N`。兩版右指都有非零接觸（原版 138/240 步、修正版 139/240 步），所以這個簡化場景**沒有重現**「原版右指全程 0 N → 修正後 279 N」。它只說明在這個場景及 PhysX 版本下接觸存在，不能否定另一個握持場景或版本的測量；要驗證那組數據，需要原來的物件、姿勢、控制與記錄程式。

## 碰撞形狀與位置

| 項目 | 舊 USD | 新 USD |
|---|---|---|
| 右手手掌 prim | `/openarm/openarm_right_hand` | `/World/openarm/follower_r_hand` |
| 右手左指 prim | `/openarm/openarm_right_left_finger` | `/World/openarm/follower_r_left_finger` |
| 右手右指 prim | `/openarm/openarm_right_right_finger` | `/World/openarm/follower_r_right_finger` |
| 左指相對手掌平移（靜態 USD） | `(0, +0.005, +0.015)` m | `(0, +0.006, +0.015)` m |
| 右指相對手掌平移（靜態 USD） | `(0, -0.005, +0.015)` m | `(0, -0.006, +0.015)` m |
| 啟用中的右手左指碰撞 prim | `/openarm/openarm_right_left_finger/collisions` | `/World/openarm/follower_r_left_finger/visuals/follower_r_left_finger_visual/Scene/mesh` |
| 啟用中的右手右指碰撞 prim | `/openarm/openarm_right_right_finger/collisions` | `/World/openarm/follower_r_right_finger/visuals/follower_r_right_finger_visual/Scene/mesh` |
| 啟用網格的頂點 | 每指 248,568 個 | 每指 248,568 個 |
| 網格近似 | `convexDecomposition`、最多 32 hulls | 同左 |
| 右手接觸 offset | 啟用 collider 上未設定 | `contactOffset=0.002 m`、`restOffset=0` |
| 左手接觸 offset | 啟用 collider 上未設定 | 沒有有限的明確設定，交由 PhysX 處理 |

兩份 USD 的**四根手指啟用碰撞網格形狀幾乎相同**：把每個網格頂點轉到各自的手指 link 座標後，以最近鄰比對，雙向最大距離約 `1.23×10⁻⁷ m`（`0.000123 mm`）。手指 link 的靜態差異是兩根各往外 1 mm，因此零位時兩指間距增加約 2 mm。這些數字沒有涵蓋 PhysX 烹煮負縮放網格時可能產生的碰撞殼錯位，也不是執行時接觸力。

對照原始 `xformOp:translate` 的 `y`：

- 右手左指：舊 `-0.158498034 m` → 新 `-0.159498036 m`。
- 右手右指：舊 `-0.148498029 m` → 新 `-0.147498026 m`。
- 手掌位置相同時，兩指因此分別往外約 1 mm。左手也有相同的 ±5 mm → ±6 mm 手掌相對位移。

用啟用網格的指尖內側邊界和上述 link 位置推算，`q=0` 的幾何間隙：舊約 `-1.523 mm`（幾何重疊），新約 `+0.477 mm`。新值也與包內 `openarm_control.py` / `PARAMS.md` 的 `jaw_gap` 公式一致。這不等同 PhysX 的實際接觸深度。

新 USD 另保留每指 792 頂點的 STL collider，但 `physics:collisionEnabled=False`，目前不參與碰撞。舊 IsaacLab USD 的啟用 collider 本來就是 248,568 頂點網格；不能把包內「舊 STL → visual mesh」敘述直接當成這兩份 USD 的差異。

## 手指驅動、材質、關節

舊關節位置：`/openarm/joints/openarm_right_finger_joint1`、`joint2`。新關節位置：`/World/openarm/joints/follower_r_finger_joint1`、`joint2`。左手相同映射：`openarm_left` → `follower_l`。

| 原始 USD 屬性 | 舊右手 joint1 | 新右手 joint1 |
|---|---:|---:|
| `drive:linear:physics:stiffness` | `1,000,000 N/m` | `327 N/m` |
| `drive:linear:physics:damping` | `100 N·s/m` | `189 N·s/m` |
| `drive:linear:physics:maxForce` | `10 N` | `333 N` |
| `drive:linear:physics:type` | `force` | `force` |
| `physxJoint:armature` | `0` | `2.75 kg` |
| `physics:lowerLimit` / `upperLimit` | `0 / 0.044 m` | `0 / 0.050 m` |

**為何配置版剛度是 327，而非 1,000,000？** 配置包的 `PARAMS.md` 與 `openarm_control.py` 把真機馬達端的比例增益 `0.1 N·m/rad`，用估算傳動比 `r=0.017495 m/rad` 換成滑動指片的線性增益：`0.1/r² ≈ 327 N/m`。這是依真機紀錄和傳動模型**推導**的值，依賴該傳動比與角度換算假設，並非直接量到的指尖剛度。配置包預期另由 `openarm_control.py` **每個物理步**給前饋，飽和廣義關節力約 `45.7 N`；只換 USD 並不會自動執行該前饋。作為量級參考，若位置誤差是 `1 mm`，`327 N/m` 的比例項只有 `0.327 N`（未計阻尼、前饋與碰撞）；舊 USD 的 `1,000,000 N/m` 比例項理論上是 `1000 N`，但原 USD 的 `maxForce=10 N` 會限制該 drive。這些都不是指片對物體的實際接觸力。**目前 lift 任務使用 `OPENARM_BI_CFG` 時，Isaac Lab actuator 另設 `stiffness=2000 N/m`，因此任務執行值不能直接用表中的原始 USD 值代表。**

從動指 `joint2`：舊沒有 position drive，armature `0`、位置範圍 `-0.0088 / 0.0528 m`；新在 USD 中有 `0.1 N/m`、`0.04 N·s/m` 的 drive 屬性，armature `2.75 kg`、位置範圍 `-0.010 / 0.060 m`。兩者均有 mimic 關係與 gearing `-1`；舊 mimic `naturalFrequency=0`、`dampingRatio=0`，新為 `120`、`0.005`。這些是 USD 靜態值，實際 PhysX 讀回可能對 mimic drive 有不同處理。

手指 link 的質量與慣量相同：每根 `physics:mass=0.03602545 kg`；對應 `physics:diagonalInertia` 也相同。

舊手指碰撞體沒有明確的 `material:binding:physics`，因此會取執行場景的預設物理材質。新四根手指綁到 `/World/GripperMaterial`：靜摩擦 `0.6`、動摩擦 `0.5`、compliant stiffness `40000 N/m`、damping `40 N·s/m`。

## 手臂與整體位姿

- 手臂關節對應路徑範例：舊 `/openarm/joints/openarm_right_joint1`、`...joint6`；新 `/World/openarm/joints/follower_r_joint1`、`...joint6`。
- 舊 USD 的 default prim 是 `/openarm`，機器人根節點在原點；新 default prim 是 `/World`，`/World/openarm` 的靜態根位姿平移 `(0, -0.475, 0.51) m` 並繞 Z 軸旋轉約 90°。不能直接比較兩份檔的世界座標；上面的手指位移都已轉成「相對手掌」。
- 右手 7 個 arm joints 的原始 USD 剛度均是 `400`；阻尼舊 `80`、新 `40`；drive type 舊 `force`、新 `acceleration`。maxForce 對應關節保持 `40/27/7`。右手 joint1 限位舊約 `[-80°, 200°]`、新 `[-135°, 135°]`；joint6 舊約 `[-45°, 45°]`、新 `[-70°, 45°]`；其餘右手關節限位基本相同。
- 原始 USD 的 articulation 兩邊皆設定 `enabledSelfCollisions=False`、solver position iterations `32`、velocity iterations `1`。執行 IsaacLab 任務時的設定會覆蓋部分值。

## 執行時設定與比較限制

`OPENARM_BI_CFG` 會把原本 USD 的夾爪改成 `stiffness=2000 N/m`、`damping=100 N·s/m`、`joint_effort_limit=333.33 N`；不能把原始 USD 的 `maxForce=10 N` 當成現在 IsaacLab 任務的實際上限。雙臂 lift 任務目前只發右手的 8 維動作，並另外覆蓋機器人初始姿勢和自碰撞設定。

配置包的 `45.7 N` 前饋是 `openarm_control.py` 每個物理步計算的努力指令，**不在 USD 內**。`1/480 s` 物理步長也是呼叫端設定。兩份 USD 的靜態差異不能直接得出「實際夾持接觸力」或「哪個模型有問題」；那要用相同物件、姿勢、材質、控制方式及時間步長跑接觸力測試。

本報告聚焦**夾持相關**差異；沒有逐節列出兩臂所有 link 的重心與慣量張量，也沒有驗證先前提到的四項官方模型錯誤是否全部成立。

在原本的 Docker 工作目錄內重跑原始屬性檢查：

```bash
cd /home/Intern_1/IsaacLab-464969
/home/Intern_1/isaaclab-venv/bin/uv run --extra isaacsim --no-sync python /home/Intern_1/openarm_usd_audit.py
```

`openarm_usd_audit.py`、`openarm_mirror_audit.py` 和 `openarm_finger_contact_ab.py` 是昨天留在掛載工作目錄的分析腳本，**不屬於這次 Markdown 所在的 Git 專案**。如果只拿 GitHub 專案到另一台機器，需先取得這些腳本，或按上列 USD prim 路徑與屬性自行核對。上述 A/B 的 `fixed` 僅修改記憶體中的 stage，沒有產生或覆寫修復後 USD。
