<div align="center">

<h1>Forward Dynamics Model</h1>

**Official implementation of “Forward Dynamics Model” (FDM).**

[🌐 Project page](https://logosroboticsgroup.github.io/FDM/) · [中文](README-zh.md)

**Learn from the future. Act in the present.**

[Overview](#overview) · [Results](#results-reported-in-the-paper) · [Installation](#installation) · [Training](#training) · [Evaluation](#evaluation-and-inference)

</div>

## Project website

The [project page](https://logosroboticsgroup.github.io/FDM/) presents the English-captioned demo, architecture comparison, training/inference pipeline, and interactive real-robot comparisons across five tasks and four methods, with all three camera views composited in each video.

For a local preview, run `python -m http.server 8000 --directory docs/website` and open `http://localhost:8000`. See [website setup and media notes](docs/website/README.md) for GitHub Pages publishing, and media provenance.

<a id="overview"></a>

## ✨ Overview

FDM learns robotic policies through **action-to-future learning**. During training, a generation expert reads action-expert features to predict future observations. The future-prediction loss backpropagates through these features, teaching the policy about the consequences of its actions. Action prediction cannot read future tokens, including through indirect context paths.

At deployment, the policy predicts actions from the current observation and task instruction, skipping future generation while retaining the context processing needed by the action expert.

<p align="center">
  <img src="docs/images/pipeline.png" alt="FDM pipeline: the context and action experts predict actions, while a training-only generation expert provides future supervision through action features." width="100%">
</p>

<p align="center"><em>Future supervision shapes action features during training; inference uses only the context and action experts.</em></p>

The same principle supports two backbone families:

| Paper model | Framework name | Backbone | Future supervision |
| --- | --- | --- | --- |
| FDM w/ VLM | `Pi05Causal` | π₀.₅ / PaliGemma | Direct regression of future-image Wan VAE latents |
| FDM w/ VGM | `WanMoTCausal` | Wan2.2-TI2V-5B | Flow matching on future-video latents |

The implementation builds on StarVLA and retains the `starVLA` Python package name. Additional Pi05 and WanMoT variants are included for related workflows; the two causal frameworks above implement FDM.

<a id="results-reported-in-the-paper"></a>

## 🏆 Results reported in the paper

Success rates (%), reproduced from the manuscript; these are paper results, not new measurements of this source release.

| Model | LIBERO Spatial | Object | Goal | Long | Average |
| --- | ---: | ---: | ---: | ---: | ---: |
| FDM w/ VLM | 99.2 | 100.0 | 99.8 | 98.6 | **99.4** |
| FDM w/ VGM | 98.8 | 100.0 | 99.2 | 94.2 | **98.1** |

| Model | RoboTwin 2.0 Clean | Randomized | Average |
| --- | ---: | ---: | ---: |
| FDM w/ VLM | 81.54 | 80.86 | **81.20** |
| FDM w/ VGM | 88.84 | 89.62 | **89.23** |

Real-world experiments use a dual-arm ARX-X5 robot on five manipulation tasks. The paper reports absolute gains of up to 13 percentage points over the compared VLA and WAM methods. Training uses eight NVIDIA H100 GPUs; inference uses one H100.

<a id="installation"></a>

## 🛠️ Installation

Run commands from the repository root. The documented environment uses Linux, Python 3.10, PyTorch 2.7.0, and CUDA 12.8.

```bash
git clone https://github.com/LogosRoboticsGroup/FDM.git
cd FDM
conda create -n fdm python=3.10 -y
conda activate fdm
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation --no-cache-dir
pip install -r requirements.txt
pip install -e .
```

Follow the source dependency instructions in [INSTALL.md](docs/INSTALL.md) to install LeRobot at commit `d602e8169cbad9e93a4a3b3ee1dd8b332af7ebf8` and build decord. Clone external repositories beside FDM. The requirements also include real-robot deployment packages; hardware drivers are needed only for the corresponding deployment workflow. Some inherited guides refer to optional environments and local tests outside this release.

<a id="data-and-pretrained-weights"></a>

## 📦 Data and pretrained weights

This release contains source code and configuration templates. Datasets, trained FDM checkpoints, text caches, and experiment logs are not bundled, and public FDM checkpoint download links are not yet provided here.

**Datasets.** Training reads LeRobot datasets. Configure dataset roots, camera keys, action dimensions, and normalization in the relevant registration:

- [LIBERO registration](examples/LIBERO/train_files/data_registry/data_config.py)
- [RoboTwin registration](examples/Robotwin/train_files/data_registry/data_config.py)
- [ARX registration](examples/ARX/train_files/data_registry/data_config.py)

For the examples below, `libero_all_multi` expects the prepared dataset under `playground/Datasets/LEROBOT_LIBERO_DATA`. The registry defines the actual dataset roots; changing a generic root field does not replace every registered path.

**FDM w/ VLM.** Prepare a compatible RLinf/OpenPI_RLinf π₀.₅ PyTorch checkpoint directory containing `model.safetensors`, the PaliGemma tokenizer, and the Wan2.2 48-channel VAE. The loader also supports the legacy OpenPI PyTorch weight layout. Set:

```bash
export PI05_MODEL_PATH=/path/to/pi05_base_pytorch
export PI05_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
```

**FDM w/ VGM.** Set the DiT, VAE, UMT5 encoder, and tokenizer paths in [starvla_wanmot_causal.yaml](starVLA/config/training/vla/starvla_wanmot_causal.yaml). The default layout uses `playground/Pretrained_models/Wan2.2-TI2V-5B` and `playground/Pretrained_models/umt5-xxl`. Training uses cached text embeddings by default:

```bash
export TEXT_EMBEDDING_CACHE_DIR=playground/cache/text_embeds/libero
PYTHONPATH=. python scripts/vla/precompute_text_embeds.py \
  --config_yaml starVLA/config/training/vla/starvla_wanmot_causal.yaml \
  --datasets.vla_data.data_mix libero_all_multi
```

Use matching encoder paths, prompt templates, and cache settings for training and serving. For a new run with a randomly initialized action expert, pass `--framework.action_model.model_path null`; the base YAML otherwise points to a local initialization checkpoint that is not distributed.

<a id="training"></a>

## 🚀 Training

### FDM w/ VLM on LIBERO

After preparing data and setting the three weight/tokenizer variables above:

```bash
NPROC_PER_NODE=8 bash examples/LIBERO/train_files/train_pi05-causal.sh \
  libero_all_multi fdm_vlm \
  --framework.model.num_images 2 \
  --trainer.resume_from_checkpoint null
```

This launcher selects a 15-step action prediction horizon, disables proprioceptive state, and trains for 30,000 steps with a global batch size of 128. The generation branch uses past/current/future frames with raw-frame offsets `[-15, 0, 15]` and a latent MSE objective. `Pi05Causal` requires FAST cross-entropy, subtask training, and RTC training delays to be disabled, as in its supplied config. See [implementation details](docs/Pi05_causal.md).

### FDM w/ VGM on LIBERO

After preparing weights and the text cache:

```bash
NPROC_PER_NODE=8 bash examples/LIBERO/train_files/train_wanmot-causal.sh \
  libero_all_multi fdm_vgm \
  --framework.action_model.model_path null \
  --datasets.vla_data.disable_state true \
  --trainer.resume_from_checkpoint null
```

The base recipe uses a 32-step action horizon, future-frame stride 4, and 22,000 optimizer steps with global batch size 128. Keep `framework.video_model.config.video_attention_mask_mode=first_frame_causal` so the current-observation features available to actions cannot read future tokens.

### RoboTwin and real robots

| Environment | FDM w/ VLM launcher | FDM w/ VGM launcher |
| --- | --- | --- |
| RoboTwin | [train_pi05-causal.sh](examples/Robotwin/train_files/train_pi05-causal.sh) | [train_WanMoTCausal.sh](examples/Robotwin/train_files/train_WanMoTCausal.sh) |
| ARX | [train_pi05-causal.sh](examples/ARX/train_files/train_pi05-causal.sh) | [train_wanmot-causal.sh](examples/ARX/train_files/train_wanmot-causal.sh) |

Launchers accept a data mixture, an optional run suffix, and trailing config overrides. `NPROC_PER_NODE` controls processes per node. Adjust per-device/global batch sizes for available hardware. Outputs go under `results/Checkpoints/vla` by default.

**Reproduction notes.** The base YAML files and launchers are starting points, and some defaults differ from the manuscript. For RoboTwin, the paper disables proprioceptive state for both models; pass `--datasets.vla_data.disable_state true`, and set VLM action/state dimensions to 14 with three camera views. For the real-world paper setup, match the 20-dimensional bimanual action representation, disabled state input, and task-specific Z-score normalization; ARX templates also support other action layouts. Match the dataset representation, camera layout, prediction horizon, and saved run configuration before comparing results. Fresh random initialization need not reproduce the exact initialization stored in the paper's runs.

<a id="evaluation-and-inference"></a>

## 🎯 Evaluation and inference

Both FDM frameworks expose action-only inference through `predict_action`; future prediction is optional through `predict_video`. Pi05Causal action inference needs only current frames and does not invoke the VAE. WanMoTCausal retains current-observation video-backbone processing while omitting future-video generation.

Install LIBERO and its evaluation dependencies following the [LIBERO guide](examples/LIBERO/README.md). Start the policy server in the model environment:

```bash
GPU_ID=0 PORT=6694 bash examples/LIBERO/eval_files/run_policy_server.sh \
  /path/to/run/final_model/pytorch_model.pt
```

In a separate terminal with the LIBERO environment activated:

```bash
PORT=6694 bash examples/LIBERO/eval_files/eval_libero.sh \
  /path/to/run/final_model/pytorch_model.pt libero_spatial
```

Repeat for `libero_object`, `libero_goal`, and `libero_10`. Keep the checkpoint's run configuration and `dataset_statistics.json` alongside the checkpoint in the expected run directory. Paths in the saved configuration must resolve on the serving machine.

Default LIBERO evaluation uses 50 episodes per task, seed 42, 10 denoising steps, and 10 executed actions before replanning. The execution horizon is distinct from the model's prediction horizon. See the [evaluation protocol](examples/eval_protocol.md), [RoboTwin guide](examples/Robotwin/README.md), and [ARX guide](examples/ARX/README.md) for the corresponding workflows.

<a id="code-structure"></a>

## 🗂️ Code structure

```text
starVLA/model/framework/VLM4A/Pi05_causal.py  # FDM w/ VLM
starVLA/model/modules/pi05/causal.py         # VLM causal experts
starVLA/model/framework/WM4A/WanMoT_causal.py # FDM w/ VGM
starVLA/model/modules/wan_mot/               # Video/action experts and MoT
starVLA/config/training/vla/                # Model and training configs
starVLA/dataloader/                         # LeRobot and video pipelines
starVLA/training/                           # Distributed training
examples/{LIBERO,Robotwin,ARX}/             # Data registrations and task workflows
deployment/                                # Policy serving and robot deployment
```

<a id="citation"></a>

## 📖 Citation

```bibtex
@misc{ma2026forward,
  title   = {Forward Dynamics Model},
  author  = {Ma, Zipei and Jiang, Junzhe and Zhang, Jiahui and Zhang, Juntong
             and Gu, Chun and Zhang, Bozhou and Deng, Jiankang and Zhu, Xiatian
             and Zhang, Li},
  year    = {2026},
  url     = {https://github.com/LogosRoboticsGroup/FDM}
}
```

<a id="acknowledgments-and-license"></a>

## 🤝 Acknowledgments and license

FDM builds on [StarVLA](https://github.com/starVLA/starVLA), with components and workflows from OpenPI/RLinf, Wan, LeRobot, LIBERO, and RoboTwin. We thank their authors and contributors. The [LICENSE](LICENSE) identifies FDM authors and contributors and retains the upstream StarVLA copyright and terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for bundled third-party attributions. Pretrained weights and datasets retain their respective licenses.
