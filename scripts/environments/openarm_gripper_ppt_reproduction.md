# OpenArm 夾爪實驗：簡報紀錄與復刻方式

來源：`2026_Automation_Taipei_參展心得_ITRI版.pptx`，第 9–14 頁。這份紀錄把簡報聲稱的結果、我們實際跑過的測試，以及尚未對齊的條件分開寫。簡報第 1 頁說明問題背景是模擬中平躺的細長器具夾不起來。

## 簡報第 9–14 頁說了什麼

| 頁 | 設定與觀察 | 解讀界線 |
| --- | --- | --- |
| 9：空夾閉合 | 不放物體、不抬手；夾爪開合指令從 50 mm 逐格到 0 mm，每格停 120 步。兩關節位置誤差及 `measured effort` 都顯示為 0。 | 簡報推測指間沒有碰撞偵測；只憑空夾的 0 N 不能證明某一指片沒有物體碰撞，也不能把它當成抓取失敗測試。 |
| 10：7 mm 棒 | 兩指之間放直徑 7 mm 的**靜態碰撞體**，逐格閉合。圖上末段的 `measured effort` 約為 joint1 0.250 N、joint2 0.013 N，兩關節位置誤差約為 5.2、6.3 mm。 | 0.250/0.013 約為 **19.2 倍**，符合頁面標題的「19 倍」；內文寫「26 倍」與這兩個數字不符。此處量的是關節量，不是左右指片對棒的接觸力。 |
| 11：單關節驅動 | 只對 joint2 下閉合指令，它仍留在約 50 mm；只驅動 joint1，joint1 閉合、joint2 也跟著閉合。 | 簡報將 joint2 解讀為沒有獨立 drive 的 follower。要驗證這點，必須同時記錄兩個關節的實際位移，並確認 joint2 的 drive 已關閉。 |
| 12：尺寸掃描 | 圓柱與方棒各測 7、15、30 mm，共 6 組；尺寸縮小時，圖上的夾持力約從 1.1 N 降到 0.5 N。簡報用有效質量 0.034 kg、stiffness 1500 與位置誤差解釋趨勢，標示 R² = 0.99999。 | 這是該模擬設定下的關節／模型力道趨勢。簡報沒有提供每組完整的物體質量、摩擦與控制紀錄，因此不能只換直徑就要求重現相同 N 數。 |
| 13：加大閉合目標 | 關閉目標超過接觸點 0、5、10、20 mm，圖上夾持力約為 0.191、0.250、0.28、0.28 N。0 mm 那組放手後滑動約 0.17 mm；超過 10 mm 後力道沒有明顯增加。 | 這是過度閉合與短時間滑動的測試，沒有呈現 5 cm 抬升成功率。 |
| 14：真機資料 | 真機 ROS bag 的兩個夾爪關節 `joint_states/effort` 都是 0；手臂其他關節有非零 effort。簡報也提到 `CommandedEEPose.gripper` 是開／合命令。 | 真機這個欄位沒有可用夾力回報；不能把它的 0 解讀成真機指片沒有接觸力，也不能用模擬 stiffness 直接推算真機夾力。 |

## 已跑過的 7 mm 對照測試

使用 `scripts/environments/openarm_thin_bar_grip_test.py`，目前任務設定使用 `OPENARM_BI_CFG` 指向的官方 `openarm_bimanual.usd`。在 `source/isaaclab_tasks/isaaclab_tasks/contrib/lift/config/openarmBi/joint_pos_env_cfg.py` 中，改用 `openarm_configured.usd` 的程式區塊目前被註解，這一組結果**不是**使用該檔。任務設定已開啟 articulation self-collision。

兩組都使用同一個直徑 **7 mm**、長 **60 mm**、質量 **30 g**、靜／動摩擦係數 **0.5** 的圓柱，種子 42、1 個環境、物理 100 Hz、控制 50 Hz。圓柱中心設於 TCP 局部 `(0.03, 0, -0.03)` m。手臂先穩定 2 s，張開 0.2 s，零重力下閉合 1.5 s；恢復重力後要求手抬高 5 cm（2 s），再保持 1 s。圓柱擺好後不再由腳本搬動。

| 控制組 | joint2 控制 | 閉合後 joint1／joint2 位置 | 閉合時左／右指片接觸力峰值 | 抬升時 joint1／joint2 驅動力估計值中位數 | 最終手／圓柱上升 |
| --- | --- | --- | --- | --- | --- |
| `both` | 兩關節都有位置目標及 drive | 4.376／4.571 mm | 4.43／4.43 N | −8.79／−9.19 N | 約 6／6 mm |
| `joint1_only` | joint2 不收 action，stiffness 與 damping 設為 0 | 4.376／4.571 mm | 4.43／4.43 N | −8.79／0 N | 約 6／6 mm |

兩份 CSV 的 235 個時間點中，兩關節實際位置、左右指片接觸力與物體位置逐列相同；改變的是 joint2 的控制目標／估算驅動力。**joint2 的 drive 估計值為 0 N 時，joint2 仍跟著移動，兩指仍有接觸力。** 這與 follower 透過機構連動的行為相符。`ImplicitActuator` 的驅動力欄位是根據控制誤差估算，不是 PhysX 直接回報的真實關節反力；負號代表閉合方向。指片接觸力來自分別過濾到圓柱的 `ContactSensorCfg`，兩者不能直接比較。

這兩組都沒有達到要求的 50 mm 抬升：手和圓柱實際只上升約 6 mm。因此可判斷本姿勢下兩指都有接觸，尚不能宣稱完成 5 cm 舉起任務，也不能把抬升不足歸因於夾爪力道。

### 在 Docker 中重跑

```bash
cd /home/Intern_1/IsaacLab-464969

# 兩個關節都有位置 drive
/home/Intern_1/isaaclab-venv/bin/uv run --extra isaacsim --no-sync python \
  scripts/environments/openarm_thin_bar_grip_test.py \
  --diameter_mm 7 --gripper_control both \
  --num_envs 1 --viz none --fast --bar_offset 0.03 0 -0.03

# 只驅動 joint1；joint2 的 drive 估計值應為 0，但請另外看實際位移與指片接觸力
/home/Intern_1/isaaclab-venv/bin/uv run --extra isaacsim --no-sync python \
  scripts/environments/openarm_thin_bar_grip_test.py \
  --diameter_mm 7 --gripper_control joint1_only \
  --num_envs 1 --viz none --fast --bar_offset 0.03 0 -0.03
```

以上是產生表格數據時的參數；要看 Isaac Sim 畫面，改成 `--viz kit` 並移除 `--fast`。

預設 CSV 分別寫入：

- `scripts/environments/grip_output/openarm_cylinder_7mm_both_grip.csv`
- `scripts/environments/grip_output/openarm_cylinder_7mm_joint1_only_grip.csv`

CSV 的 `finger_joint*_m` 是實際關節位移；`finger_joint*_target_m` 是位置目標；`finger_joint*_drive_est_N` 是估算 drive；`left_normal_N`／`right_normal_N` 才是指片對圓柱的接觸力。重跑會覆寫同名 CSV；想保留新試次，可加 `--csv <另一個檔名>`。

## 若要更貼近簡報原實驗

目前腳本**只對齊 7 mm 直徑與單關節控制比較**。我們的圓柱是可受重力影響的動態剛體，會進行抬升；簡報第 10 頁放的是靜態碰撞體，並逐格停留。要重現第 9～13 頁的方法，應另外建立下列測試，不可把上表數值當成與簡報的直接對照：

1. **空夾（第 9 頁）**：移除圓柱，固定手臂姿勢；夾爪目標按 50、40、30、20、10、0 mm 遞減，每格停 120 個物理步。先確認目前 USD 的張開上限能否達到 50 mm；若上限只有 44 mm，要記錄這個差異，不能默默截斷後說是相同條件。
2. **靜態 7 mm 棒（第 10～11 頁）**：在兩指間放固定的 7 mm 圓柱，確認張開時不穿模；用可連續指定位置目標的控制（現有 binary action 不支援逐格目標），每格保持 120 步，尤其記錄 10、5、0 mm 附近。分別執行 `both` 與 `joint1_only`，記錄兩關節目標、實際位置、速度、估算 drive 及左右指片對棒的接觸力。
3. **尺寸掃描（第 12 頁）**：只改圓柱／方棒截面為 7、15、30 mm，其他質量、摩擦、長度、初始位置、控制參數及取樣時間保持一致；每組都留原始逐步資料，而不只留峰值。
4. **過度閉合（第 13 頁）**：找出第一次接觸的目標，再加 0、5、10、20 mm 的閉合量；每組都記錄是否碰到關節行程限制。依簡報流程，在夾住後等待 1 s 再釋放支撐，觀察 5 s 的相對滑移與左右接觸力。

簡報未附原始程式、USD 版本、棒長度、質量／摩擦、精確擺位及 `measured effort` 的讀取 API。這些條件未確認前，可以復刻**操作流程和要觀察的現象**，不能保證復現 0.250 N、0.013 N 或圖上的曲線。
