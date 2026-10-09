# Pi05Causal

`starVLA/model/framework/VLM4A/Pi05_causal.py` 注册 `Pi05Causal`。沿用 InternA1Causal 的直接 latent 回归设计：

- understand：SigLIP 编码当前帧 `o_t`，加语言及原 Pi05 的离散 state 输入。
- action：保留 Pi05 flow matching 和推理时的 understand KV cache。
- generation：输入 `o_{t-15}` 与 `o_t` 的 Wan2.2 latent，回归 `o_{t+15}` latent。
  每层独立初始化的 Q/K/V/O、MLP、adaRMSNorm 与 action expert 使用相同配置；不复制 action 权重。
  输入/输出投影适配 48 通道 latent。generation 是直接回归，adaRMS 使用独立 MLP 编码固定 `t=0`。
- 分块注意力顺序为 understand → action → generation，各块内双向。
  generation 能读取 noisy actions；action 不能读取 generation。Pi05 的 state 在 understand 中，无额外 state token。
- 冻结 Wan2.2 VAE，逐帧以 `T=1` 独立编码，空间降采样 16 倍。
  latent 先经 1×1 卷积映射到 expert 宽度，再经 kernel=stride=`scale_factor` 的卷积压缩空间 token。
  默认 256×256 图像产生 16×16 Wan latent，`scale_factor=4` 将其压到 4×4；每帧每视角 16 token，
  两帧三视角共 96 generation token（压缩前为 1536）。这与 InternA1 的 32×32 Cosmos latent 经 8 倍压缩
  所得到的 token 数一致；这里是可学习的步长卷积，不是 AvgPool。
  对 expert 输出的两帧特征按对应空间位置取均值，再经转置卷积还原到完整 latent 网格，
  经 LayerNorm 和输出投影回归未来的 48 通道 latent。
  训练只计算有效 camera 的 latent MSE，不经过像素解码。

基础 Pi05 checkpoint 先严格加载，再添加随机 generation expert。完整 causal checkpoint 由现有框架恢复接口加载；
`framework.model.model_path` 应指向基础 Pi05 权重，不能指向 causal checkpoint。
原 `Pi05` 配置、权重命名及动作推理行为保持兼容。新模型仍使用原 Pi05 的 action loss 维度归约方式。
本次迁移保留当前 Pi05 的 FAST、subtask 和 RTC 实现；`Pi05Causal` 延续源实现的 flow + generation
训练目标，默认关闭 FAST CE。设置非零 `action_ce_weight`、启用 `subtask_enabled` 或非零
`rtc_training_max_delay` 会明确报错，避免配置被静默忽略。推理仍可使用继承的 RTC 动作接口。

`scale_factor` 必须是正整数且整除 `image_size / 16`。此次空间压缩修正增加了卷积参数，
并将 generation 输入 Linear 改为 Conv2d，因此早期无压缩版本的 causal checkpoint 不能直接严格加载。

## 数据与推理

示例配置：`starVLA/config/training/vla/starvla_pi05_causal.yaml`。
两个 LeRobot loader 均支持 `image_frame_offsets: [-15, 0, 15]`，以原始视频帧计数，
超出 episode 的索引截到边界；过滤静止帧后也不改变这个时间单位。未配置 offsets 的加载行为保持不变。

每个样本的 `image` 是 camera 列表，每个 camera 为 `[3,3,H,W]`，时间顺序是过去、当前、未来。
也可提供 `past_image`、`image`、`future_image` 三个 camera 列表。
`view_mask` 与 camera 列表对齐，缺少的 camera slot 自动补零并屏蔽。

`predict_action(examples=...)` 接受当前帧或上述时序帧；只使用当前帧，无需 VAE。
`predict_video(examples=..., actions=...)` 接受过去/当前两帧（或训练的三帧，忽略未来）。
`actions` 为归一化的 `[B,T,action_dim]`，取最后 action_horizon 步；不提供则先采样动作。
推理 generation 读取最终动作，action timestep 固定为 0。
返回 `video: [B,V,3,1,H,W]`、范围 `[0,1]`，以及 `normalized_actions`。
视频推理必须提供过去帧；在 episode 起点可明确将当前帧同时作为过去帧。

## 运行与验证

先把配置中的 `model_path`、`tokenizer_path`、`vae_path`、数据路径替换为本地路径。基础权重使用现有
Pi05 支持的 RLinf/OpenPI checkpoint；VAE 使用 Wan2.2 48-channel checkpoint（`.pth` 或 `.safetensors`）。

```bash
accelerate launch --num_processes 8 starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/vla/starvla_pi05_causal.yaml

TORCHDYNAMO_DISABLE=1 python -m unittest \
  scripts.test.test_pi05_causal \
  scripts.test.test_pi05_causal_video_gt \
  scripts.test.test_pi05_causal_vae_batch \
  scripts.test.test_pi05_state_input \
  scripts.test.test_pi05_training_time_rtc \
  scripts.test.test_pi05_fast_loss \
  scripts.test.test_pi05_subtask -v
```

本地回归脚本按仓库规则放在被忽略的 `scripts/test/`，不纳入提交。
2026-10-08 在 `starVLA` conda 环境运行上述测试：47 项，46 项通过、1 项跳过
（缺少本地 LeRobot 参考源码，无法进行 RTC parity 对照）。为执行 CPU 检查设置
`TORCHDYNAMO_DISABLE=1`，未验证 torch.compile 编译路径。

迁移测试覆盖原权重保持、随机 expert、因果隔离、KV cache 数值一致性、camera mask、
activation checkpointing 反向传播、latent 输出尺寸、当前帧选择、视频推理接口、VAE 按时间独立编码、
未来 GT 选择和原始帧偏移；并运行现有 Pi05 的 state、RTC、FAST 和 subtask 回归测试。
小型模型和替身 codec 的测试不替代预训练权重与真实 Wan2.2 编解码验证。

新增模型文件及修改的 `Pi05.py` 通过 `black --check` 和 `ruff check`。
两个 loader 在 HEAD 基线均已有 Black 格式问题，Ruff 分别为 33、9 项；本次检查数量及问题类型不变，
未作无关全文件格式化。`.gitignore` 的 `scripts/test/*` 规则保持不变。

当前环境无可用 GPU，未运行完整 GPU 训练、预训练 Pi05 parity、真实 Wan2.2 编解码或 benchmark。
尚无本次迁移的 benchmark 结果和公开 Pi05Causal checkpoint；这两项仍是发布前的验证缺口。
