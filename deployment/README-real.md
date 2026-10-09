# 0. 有用小工具

1. 回到 home pose
```bash
cna starVLA
python -m deployment.teleoperation.gohome --config examples/Piper/configs/piper_vr_dual.yaml
```
2. 拍摄3个摄像头
```bash
cna starVLA
python -m deployment.teleoperation.camera
```

# 1. 启动遥操作

```bash
cna starVLA
python -m deployment.teleoperation.cli teleoperate --config examples/Piper/configs/piper_vr_dual.yaml
```

# 2. LeRobot 录制

```bash
DATASET_ROOT="examples/Piper/recordings/piper_vr_$(date +%Y%m%d_%H%M%S)"

python -m deployment.teleoperation.cli record \
  --config examples/Piper/configs/piper_vr_dual.yaml \
  --root "$DATASET_ROOT" \
  --repo-id local/piper_vr \
  --task "Pick up the object and place it in the target area"
```

- 启动后双臂先回 home；按左手柄 `X` 显式开始录制，trigger 只负责遥操作。
- 右手柄 `A`：松开双手 trigger 后立即停止采帧，双臂先回 home，再等待当前 episode 保存完成。
- 右手柄 `B`：松开双手 trigger 后立即停止采帧，双臂先回 home，再丢弃当前 episode。
- 键盘只保留 `q`/Esc 退出；左 `X` 开始，右 `A` 保存，右 `B` 丢弃。
- `q`/Esc：退出并丢弃尚未保存的当前 episode。
- 续采已有数据集时追加 `--resume`。
- 当前三相机配置在回 home 后使用 `video_encoding_workers: 3` 并行编码；保存耗时见终端
  `Recording episode saved: ... elapsed_s=...`。

完整相机配置、字段 schema 和注意事项见 `examples/Piper/README.md`。

# 3. 校验和回放

默认只做完整回放预检，不连接机械臂：

```bash
python -m deployment.teleoperation.cli replay \
  --config examples/Piper/configs/piper_vr_dual.yaml \
  --root "$DATASET_ROOT" \
  --repo-id local/piper_vr \
  --episode 0 \
  --validate-images
```

确认环境已复位后，第一次建议半速真机回放：

```bash
python -m deployment.teleoperation.cli replay \
  --config examples/Piper/configs/piper_vr_dual.yaml \
  --root "$DATASET_ROOT" \
  --repo-id local/piper_vr \
  --episode 0 \
  --speed 0.5 \
  --max-joint-step-deg 16 \
  --execute
```

只有 `--execute` 会使能双臂。回放前后默认回 home；`Ctrl-C` 中断时只停止下发和断开 CAN，不会自动回 home。
`--max-joint-step-deg 16` 设置最大角度。



# 采集记录

-0829 目标采集100-150条排水管插入烧杯数据
