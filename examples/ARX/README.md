# 双 ARX X5：VR 遥操作、录制与推理

复用 `examples/Piper` 的 WebXR、相机、LeRobot episode 控制和同步推理链路。
硬件驱动使用 [real-stanford/arx5-sdk](https://github.com/real-stanford/arx5-sdk) 的
`arx5_interface`，不是 ARX 官方 `arx_x5_python` 接口。两台 X5 分别使用独立 CAN 接口。
不使用仓库其他 `arx_x5` 数据或训练配置。

## 环境和机器配置

在现有 StarVLA 部署环境安装 `pip install -r examples/ARX/requirements.txt`；VR 依赖 draccus、numpy、scipy、
aiohttp、websockets，推理另需 pyzmq、OpenCV，录制另需本项目兼容的 LeRobot 环境。
SDK 源码安装与 CAN 配置参考上游 README。源码 checkout 放在本仓库旁边。

所有启动脚本默认共用 **`configs/arx_vr_dual.yaml`**，不再维护独立录制或推理 YAML。
当前文件包含这台双臂机器的 CAN、夹爪标定、home、TLS 路径和相机序列号；换机器时需重新核对：

- 左右臂 `can_interface`，确保对应各自的物理机械臂。
- 两臂 `home_joints_deg`、`arx.home_duration_s`；当前为用户确认的遥操 A 键 home。
- `gripper_max_width_m` 和 `arx.gripper_open_readout`。示例宽度来自 SDK 的 X5 默认值，
  不是这台机器的标定结果；不同夹爪版本可能需要负的 readout。
- `tool_from_j6_xyz_mm_rpy_deg` 沿用共享字段名，但在 ARX 中表示 **SDK URDF 的 eef_link 到 TCP**。
  零变换使用 SDK 末端。`ik_urdf_path` 可覆盖 SDK URDF；底座和末端 link 名仍采用 SDK X5 配置。
- VR 的 bind 地址、TLS 证书、允许的 origin、坐标映射；相机路径或 RealSense 序列号。
  相机只在 `recording.cameras` 定义一次，推理自动映射为 `observation.images.<name>`。

ARX 使用 SDK 的关节限位、速度/力矩保护和控制增益，以及同步 host IK。
Piper 的 `move_speed_percent`、`home_move_speed_percent`、`gripper_effort`、
`gripper_stall_*` 和 IK worker 设置不参与 ARX 控制；回 home 通过 `arx.home_duration_s` 插值。
SDK 夹爪过流保护由 SDK 执行。
反向夹爪（`gripper_open_readout < 0`）需要上游修复后的 SDK：x86_64 使用 `0.1.3`，
aarch64 使用 `0.1.4`。旧版本会把夹爪受阻保护应用到错误方向，主机减压逻辑不能替代这个修复。
参见 [上游更新说明](https://github.com/real-stanford/arx5-sdk#update-20260715)。
夹爪位置增益采用采集工作台的 `kp=2.0`。闭合受阻且反馈力矩达到 SDK 上限的 50% 时，
将目标设为当前开度加 1.5 mm 并锁住继续闭合；发出比锁定开度大 1 mm 的张开目标后解除锁定。
该处理也覆盖松开 trigger 后仍未完成的闭合目标；录制使用实际发送的开度。
这只是主机控制循环中的减压处理，不替代 SDK 过流保护，也不保证物体不会松脱。
另通过 `arx.gripper_velocity_limit_m_s` 将 SDK 夹爪速度限制为默认 0.04 m/s，
通过 `arx.gripper_max_closing_error_m` 将闭合目标限制在实测开度以内最多 4 mm。
位置误差约束以反馈为基准，连续闭合指令不会在夹住物体后继续积累目标误差；
减压、限速与误差约束均不能保证消除冲击过流，需实机验证。旧本机配置未填写这两项时采用上述默认值。

若复用本机 `localhost:9527` 采集工作台的标定，在该页面浏览器的
`localStorage["collector.robotConfig.arx5"]` 中读取左右臂的 `gripper_open_readout`，
核对 CAN 接口后写入统一 YAML 的 `arms.<side>.arx.gripper_open_readout`。
保留读数的正负号，不要混用 `arx5_piper` 的另一组配置。
该标定先在闭合位置写入电机零点，再手动全开测量带符号读数；复制 YAML 参数不会重设电机零点。
证书和私钥文件仍放在被 Git 忽略的 `secrets/`，YAML 只保存路径。

## 先验证，再运行

当前配置启用了硬件与运动。以下配置校验命令不连接机械臂：

```bash
python -m deployment.teleoperation.cli validate-config --config examples/ARX/configs/arx_vr_dual.yaml
python -m deployment.robot_inference.arx_cli validate-config --config examples/ARX/configs/arx_vr_dual.yaml
```

使用模式由命令选择：teleop 只遥操，record 开启同步录制。所有脚本启动时打印实际配置路径。
无需再设置 `CONFIG`；如终端曾导出旧路径，先执行 `unset CONFIG`。

```bash
bash examples/ARX/run_teleop.sh
bash examples/ARX/run_record.sh \
  --root results/arx_vr/session_001 --repo-id local/arx_vr --task 'pick up the object'
```

VR 操作与 Piper 一致：trigger 按住后建立相对位姿锚点，squeeze 控制夹爪；
按住任一 trigger 自动开始 episode（也支持左手柄 X），右手柄 A 保存并回 home，B 丢弃并回 home，q/Esc 退出。
遥操作中需松开两个 trigger 再按 A 回 home。录制启动会先回 home。
失去追踪或 VR 超时后不再提交新位姿目标，SDK 继续保持最后目标。
退出时先将两臂置为阻尼状态，再释放控制器；不会自动回 home。

SDK 构造控制器会发电机指令，因此仅在运动开关开启后创建。ARX 不支持共享 CLI 的
`probe`、`ik-preview` 或 Piper 独立 gohome 工具，避免将 SDK 初始化误称为只读操作。

## 回放录制数据

回放发送数据集的 `action.joint`，包含双臂关节弧度和归一化夹爪开度，不启动 VR 或相机。
使用与录制相同的 CAN、夹爪标定和 home 配置。先退出遥操或录制程序，将物体恢复到演示起点。

```bash
# 仅检查数据、起点、逐帧关节变化和 SDK 关节范围，不创建电机控制器
bash examples/ARX/run_replay.sh \
  --root results/arx_vr/session_001 --episode 0

# 实机半速回放；--episode 1 选择第二段
bash examples/ARX/run_replay.sh \
  --root results/arx_vr/session_001 --episode 0 --speed 0.5 --execute
```

实机启动先回 home，完成后再次回 home，然后进入阻尼。`--no-return-home` 只跳过结束回 home，
仍会进入阻尼；Ctrl+C 中止时也不回 home。速度参数只缩放回放时间，不改变 home 速度。
回放保留 SDK 保护和夹爪受阻处理，因此实际轨迹、夹持力度可能与演示不同。

## 推理

使用本次 ARX 录制格式训练的 checkpoint 启动现有 model server，保证 `stat_key`、
`action_space`、维度、相机顺序与 统一 YAML 的 `inference` 部分 一致。随后：

```bash
python -m deployment.robot_inference.arx_cli server-metadata --config examples/ARX/configs/arx_vr_dual.yaml
bash examples/ARX/run_client.sh
```

推理先校验服务端 metadata、连接相机，再启动机械臂并回 home；VR 下用 X 开始推理 episode，A 保存并回 home，B 丢弃并回 home。
也可通过 `--controls keyboard` 使用共享键盘控制。推理是否记录由统一配置的 `recording.enabled` 决定，
示例中已开启，运行前将 `recording.root` 指向新的推理输出目录，避免覆盖遥操数据集。
启动脚本默认添加 `--execute`；可通过 `CONFIG`、`CONTROLS`、`SERVER_IP`、`PORT`、`STAT_KEY`、
`PROMPT`、`CAMERA_ORDER`、`RECV_TIMEOUT_MS` 和 `N_EXECUTE` 环境变量覆盖客户端参数。

## 数据约定

| 字段 | 左臂在前，右臂在后，每臂 7 维 |
| --- | --- |
| `state.joint` / `action.joint` | `[q1..q6 (rad), gripper_open]` |
| `state.eef` / `action.eef` | `[x,y,z (mm), rotation_vector_xyz (rad), gripper_open]` |
| `action.executed` | 每臂是否提交了位姿/关节目标；推理中也包括独立夹爪指令 |

夹爪 0 为闭合、1 为完全打开；与 SDK 的米制宽度双向换算。
EEF 姿态是旋转向量，不是 SDK 的 RPY。位姿在各自底座坐标系中。
`state` 来自反馈，`action` 是提交给 SDK 的目标，SDK 内部插值/限速后的瞬时输出可能不同。
IK 失败不提交新的关节目标，独立夹爪仍可工作；动作记录保留此前目标。

遥操作数据的 `robot_type=dual_arx_x5_vr`，推理数据为 `dual_arx_x5_inference`；
不能 resume 到 Piper 数据集。训练复用 Piper 数据配置；硬件类型不同，不代表 Piper checkpoint 可直接用于 ARX。

## 训练

`train_files/data_registry/data_config.py` 注册两种 mixture，直接引用现有 Piper 的 `data_type`，
不复制数据适配器。三路图像顺序为 third_view、left_wrist、right_wrist。

| data_mix | data_type / 推理 stat_key | 状态与动作 | 推理 action_space |
| --- | --- | --- | --- |
| `arx_vr` | `piper` | `state.eef` / `action.eef`，复用 Piper 相对 EEF 变换 | `cartesian` |
| `arx_vr_joint` | `piper_joint` | `state.joint` / `action.joint`，绝对关节位置 | `joint_position` |

激活 StarVLA 训练环境后，从仓库根目录执行：

```bash
ARX_DATA_ROOT=/path/to/arx_recording \
PI05_MODEL_PATH=/path/to/pi05_base_pytorch \
bash examples/ARX/train_files/train_pi05.sh arx_vr experiment

# 关节空间训练
ARX_DATA_ROOT=/path/to/arx_recording \
PI05_MODEL_PATH=/path/to/pi05_base_pytorch \
bash examples/ARX/train_files/train_pi05.sh arx_vr_joint experiment
```

`ARX_DATA_ROOT` 默认为 `results/arx_vr`，须指向包含 `meta/info.json` 的单个 LeRobot 数据集。
多个 session 可在本目录的 registry 中增加配方。训练脚本复用公共 Pi0.5 YAML，支持
`NPROC_PER_NODE`、`RUN_ID` 和尾部配置覆盖；模型、tokenizer 路径按本地环境设置。
关节训练后，推理 YAML 同时设置 `stat_key: piper_joint` 和 `action_space: joint_position`。
统计来自当前 ARX 训练数据；复用的是数据结构和变换，不是 Piper 数据集的统计数值。
此处针对本示例录制的双臂格式，不适用于 `scripts/data/convert_arx_to_lerobot.py` 的旧 NAS 格式。

## 验证范围

配置加载、训练注册和无硬件回归可在本地执行。实际双臂、VR 头显、相机、
GPU 训练和 checkpoint 推理仍需在目标设备验证；尚未提供本次集成的真机评测结果或公开 checkpoint。
本地回归脚本保存在 `scripts/test/`，按仓库规则不提交：

```bash
python -m unittest scripts.test.test_arx_integration -v
```
