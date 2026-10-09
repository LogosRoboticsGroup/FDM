# 🚀 RoboTwin 2.0 评测

本文介绍如何在 [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin) 上复现我们的**实验结果**。  
评测流程主要分为两部分：  

1. 配置 `robotwin-star` 环境并安装依赖。  
2. 在 `starVLA` 和 `robotwin-star` 环境中分别启动服务，运行评测。  

我们已验证此流程可在 **NVIDIA 4090** GPU 上成功运行。  

# 数据集下载

StarVLA 在 Hugging Face 上发布了 LeRobot 格式的训练数据。[StarVLA/RoboTwin-Randomized](https://huggingface.co/datasets/StarVLA/RoboTwin-Randomized) 仓库**同时包含 Clean 和 Randomized 数据**，分别位于两个顶层目录中：

| 子集 | 仓库目录 | 示范数据量 |
|------|----------|------------|
| Clean | [Clean/](https://huggingface.co/datasets/StarVLA/RoboTwin-Randomized/tree/main/Clean) | 50 项任务 × 每项任务 50 条轨迹 |
| Randomized | [Randomized/](https://huggingface.co/datasets/StarVLA/RoboTwin-Randomized/tree/main/Randomized) | 50 项任务 × 每项任务 500 条轨迹 |

Hugging Face 当前显示完整仓库大小约为 **79.6 GB**。以下命令均在 StarVLA 仓库根目录执行。

## 下载两种数据

激活用于下载的环境，并检查 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) 是否可用：

```bash
conda activate robotwin-star
hf --help
```

如果没有 `hf` 命令，可运行 `python -m pip install "huggingface_hub>=0.34,<1.0"` 安装。

```bash
hf download StarVLA/RoboTwin-Randomized \
    --repo-type dataset \
    --local-dir playground/Datasets/RoboTwin
```

下载后的目录结构如下：

```text
playground/Datasets/RoboTwin/
├── Clean/
│   ├── adjust_bottle/
│   │   ├── data/
│   │   ├── meta/
│   │   └── videos/
│   └── ...
└── Randomized/
    ├── adjust_bottle/
    │   ├── data/
    │   ├── meta/
    │   └── videos/
    └── ...
```

远端仓库已经包含 `Clean/` 和 `Randomized/`，因此 `--local-dir` 应设置为两者共用的 `RoboTwin` 根目录。该结构与 [RoboTwin 数据注册文件](train_files/data_registry/data_config.py)中的路径一致。

# 实验结果


<details open>
<summary><b>RoboTwin 2.0 的 50 项任务基准测试结果（扩大数据规模的设置）</b></summary>

### 训练数据集

模型使用官方 **RoboTwin 2.0 数据集**进行训练。

* 无随机化示范数据：50 项任务 × 每项任务 50 条轨迹
* 随机化示范数据：50 项任务 × 每项任务 500 条轨迹

| 任务名称 | StarVLA-OFT 简单 | StarVLA-OFT 困难 | π0 简单 | π0 困难 | π0.5 简单 | π0.5 困难 | X-VLA 简单 | X-VLA 困难 | Motus 简单 | Motus 困难 | lingbot-vla 不使用深度 简单 | lingbot-vla 不使用深度 困难 | lingbot-vla 使用深度 简单 | lingbot-vla 使用深度 困难 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Adjust Bottle | 100 | 99 | 99 | 95 | 100 | 99 | 100 | 99 | 89 | 93 | 100 | 100 | 100 | 100 |
| Beat Block Hammer | 93 | 92 | 79 | 84 | 96 | 93 | 92 | 88 | 95 | 88 | 87 | 91 | 92 | 89 |
| Blocks Ranking RGB | 99 | 98 | 80 | 63 | 92 | 85 | 83 | 83 | 99 | 97 | 92 | 91 | 92 | 91 |
| Blocks Ranking Size | 79 | 80 | 14 | 5 | 49 | 26 | 67 | 74 | 75 | 63 | 66 | 73 | 76 | 70 |
| Click Alarmclock | 58 | 51 | 77 | 68 | 98 | 89 | 99 | 99 | 100 | 100 | 93 | 26 | 97 | 43 |
| Click Bell | 23 | 27 | 71 | 48 | 99 | 66 | 100 | 100 | 100 | 100 | 32 | 19 | 43 | 36 |
| Dump Bin Bigbin | 91 | 94 | 88 | 83 | 92 | 97 | 79 | 77 | 95 | 91 | 97 | 92 | 97 | 97 |
| Grab Roller | 100 | 100 | 98 | 94 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 99 | 100 | 100 |
| Handover Block | 97 | 93 | 47 | 31 | 66 | 57 | 73 | 37 | 86 | 73 | 80 | 83 | 83 | 95 |
| Handover Mic | 98 | 96 | 97 | 97 | 98 | 97 | 0 | 0 | 78 | 63 | 94 | 98 | 94 | 99 |
| Hanging Mug | 34 | 29 | 14 | 11 | 18 | 17 | 23 | 27 | 38 | 38 | 32 | 27 | 34 | 53 |
| Lift Pot | 100 | 100 | 80 | 72 | 96 | 85 | 99 | 100 | 96 | 99 | 100 | 99 | 100 | 100 |
| Move Can Pot | 91 | 90 | 68 | 48 | 51 | 55 | 89 | 86 | 34 | 74 | 79 | 84 | 89 | 87 |
| Move Pillbottle Pad | 98 | 100 | 67 | 46 | 84 | 61 | 73 | 71 | 93 | 96 | 93 | 94 | 92 | 90 |
| Move Playingcard Away | 100 | 98 | 74 | 65 | 96 | 84 | 93 | 98 | 100 | 96 | 96 | 99 | 98 | 100 |
| Move Stapler Pad | 74 | 90 | 41 | 24 | 56 | 42 | 78 | 73 | 83 | 85 | 74 | 49 | 74 | 48 |
| Open Laptop | 98 | 100 | 71 | 81 | 90 | 96 | 93 | 100 | 95 | 91 | 96 | 96 | 98 | 96 |
| Open Microwave | 28 | 39 | 4 | 32 | 34 | 77 | 79 | 71 | 95 | 91 | 91 | 75 | 91 | 92 |
| Pick Diverse Bottles | 87 | 86 | 69 | 31 | 81 | 71 | 58 | 36 | 90 | 91 | 79 | 86 | 88 | 85 |
| Pick Dual Bottles | 91 | 93 | 59 | 37 | 93 | 63 | 47 | 36 | 96 | 90 | 82 | 95 | 99 | 90 |
| Place A2B Left | 90 | 95 | 43 | 47 | 87 | 82 | 48 | 49 | 82 | 79 | 86 | 83 | 89 | 85 |
| Place A2B Right | 88 | 95 | 39 | 34 | 87 | 84 | 36 | 36 | 90 | 87 | 74 | 77 | 80 | 80 |
| Place Bread Basket | 91 | 78 | 62 | 46 | 77 | 64 | 81 | 71 | 91 | 94 | 92 | 93 | 95 | 93 |
| Place Bread Skillet | 89 | 80 | 66 | 49 | 85 | 66 | 77 | 67 | 86 | 83 | 90 | 89 | 90 | 92 |
| Place Burger Fries | 100 | 100 | 81 | 76 | 94 | 87 | 94 | 94 | 98 | 98 | 95 | 96 | 98 | 94 |
| Place Can Basket | 75 | 75 | 55 | 46 | 62 | 62 | 49 | 52 | 81 | 76 | 68 | 78 | 75 | 72 |
| Place Cans Plasticbox | 100 | 99 | 63 | 45 | 94 | 84 | 97 | 98 | 98 | 94 | 97 | 100 | 100 | 98 |
| Place Container Plate | 99 | 99 | 97 | 92 | 99 | 95 | 97 | 95 | 98 | 99 | 99 | 99 | 99 | 100 |
| Place Dual Shoes | 91 | 89 | 59 | 51 | 75 | 75 | 79 | 88 | 93 | 87 | 80 | 83 | 87 | 86 |
| Place Empty Cup | 100 | 100 | 91 | 85 | 100 | 99 | 100 | 98 | 99 | 98 | 100 | 100 | 100 | 100 |
| Place Fan | 94 | 95 | 66 | 71 | 87 | 85 | 80 | 75 | 91 | 87 | 91 | 79 | 92 | 87 |
| Place Mouse Pad | 87 | 94 | 20 | 20 | 60 | 39 | 70 | 70 | 66 | 68 | 82 | 78 | 86 | 79 |
| Place Object Basket | 93 | 94 | 67 | 70 | 80 | 76 | 44 | 39 | 81 | 87 | 90 | 91 | 90 | 88 |
| Place Object Scale | 93 | 93 | 57 | 52 | 86 | 80 | 52 | 74 | 88 | 85 | 84 | 90 | 90 | 88 |
| Place Object Stand | 99 | 98 | 82 | 68 | 91 | 85 | 86 | 88 | 98 | 97 | 97 | 93 | 93 | 88 |
| Place Phone Stand | 86 | 95 | 49 | 53 | 81 | 81 | 88 | 87 | 87 | 86 | 92 | 93 | 90 | 87 |
| Place Shoe | 96 | 100 | 76 | 76 | 92 | 93 | 96 | 95 | 99 | 97 | 99 | 94 | 99 | 99 |
| Press Stapler | 99 | 96 | 44 | 37 | 87 | 83 | 92 | 98 | 93 | 98 | 90 | 88 | 86 | 93 |
| Put Bottles Dustbin | 90 | 85 | 65 | 56 | 84 | 79 | 74 | 77 | 81 | 79 | 88 | 92 | 92 | 93 |
| Put Object Cabinet | 89 | 91 | 73 | 60 | 80 | 79 | 46 | 48 | 88 | 71 | 92 | 86 | 85 | 88 |
| Rotate QRcode | 88 | 90 | 74 | 70 | 89 | 87 | 34 | 33 | 89 | 73 | 93 | 84 | 86 | 82 |
| Scan Object | 94 | 91 | 55 | 42 | 72 | 65 | 14 | 36 | 67 | 66 | 91 | 97 | 92 | 96 |
| Shake Bottle Horizontally | 100 | 100 | 98 | 92 | 99 | 99 | 100 | 100 | 100 | 98 | 100 | 100 | 99 | 98 |
| Shake Bottle | 100 | 100 | 94 | 91 | 99 | 97 | 99 | 100 | 100 | 97 | 99 | 100 | 100 | 99 |
| Stack Blocks Three | 94 | 86 | 72 | 52 | 91 | 76 | 6 | 10 | 91 | 95 | 92 | 99 | 96 | 95 |
| Stack Blocks Two | 100 | 100 | 93 | 79 | 97 | 100 | 92 | 87 | 100 | 98 | 100 | 100 | 100 | 99 |
| Stack Bowls Three | 95 | 91 | 77 | 75 | 77 | 71 | 76 | 86 | 79 | 87 | 72 | 83 | 71 | 77 |
| Stack Bowls Two | 99 | 100 | 94 | 95 | 95 | 96 | 96 | 93 | 98 | 98 | 92 | 95 | 90 | 97 |
| Stamp Seal | 86 | 90 | 46 | 33 | 79 | 55 | 76 | 82 | 93 | 92 | 76 | 86 | 74 | 77 |
| Turn Switch | 65 | 62 | 41 | 42 | 62 | 54 | 40 | 61 | 84 | 78 | 61 | 65 | 67 | 63 |
| **平均值** | **88.18** | **88.32** | **65.92** | **58.40** | **82.74** | **76.76** | **72.80** | **72.84** | **88.66** | **87.02** | **86.50** | **85.34** | **88.56** | **86.68** |

*注：全部 50 项任务在同一个模型中训练，每项任务使用 50 条无随机化示范和 500 条随机化示范进行联合训练。模型检查点可从 [Qwen3-VL-OFT-Robotwin2-All](https://huggingface.co/StarVLA/Qwen3-VL-OFT-RoboTwin2-All) 下载。*

</details>


<details close>
<summary><b>RoboTwin 2.0 的 50 项任务基准测试结果</b></summary>


| 任务名称 | RDT 简单 | RDT 困难 | Pi0 简单 | Pi0 困难 | ACT 简单 | ACT 困难 | DP 简单 | DP 困难 | DP3 简单 | DP3 困难 | StarVLA-OFT 简单 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Adjust Bottle | 81 | 75 | 90 | 56 | 97 | 23 | 97 | 0 | 99 | 3 | 96 |
| Beat Block Hammer | 77 | 37 | 43 | 21 | 56 | 3 | 42 | 0 | 72 | 8 | 58 |
| Blocks Ranking RGB | 3 | 0 | 19 | 5 | 1 | 0 | 0 | 0 | 3 | 0 | 45 |
| Blocks Ranking Size | 0 | 0 | 7 | 1 | 0 | 0 | 1 | 0 | 2 | 0 | 27 |
| Click Alarmclock | 61 | 12 | 63 | 11 | 32 | 4 | 61 | 5 | 77 | 14 | 91 |
| Click Bell | 80 | 9 | 44 | 3 | 58 | 3 | 54 | 0 | 90 | 0 | 94 |
| Dump Bin Bigbin | 64 | 32 | 83 | 24 | 68 | 1 | 49 | 0 | 85 | 53 | 68 |
| Grab Roller | 74 | 43 | 96 | 80 | 94 | 25 | 98 | 0 | 98 | 2 | 93 |
| Handover Block | 45 | 14 | 45 | 8 | 42 | 0 | 10 | 0 | 70 | 0 | 0 |
| Handover Mic | 90 | 31 | 98 | 13 | 85 | 0 | 53 | 0 | 100 | 3 | 39 |
| Hanging Mug | 23 | 16 | 11 | 3 | 7 | 0 | 8 | 0 | 17 | 1 | 15 |
| Lift Pot | 72 | 9 | 84 | 36 | 88 | 0 | 39 | 0 | 97 | 0 | 0 |
| Move Can Pot | 25 | 12 | 58 | 21 | 22 | 4 | 39 | 0 | 70 | 6 | 50 |
| Move Pillbottle Pad | 8 | 0 | 21 | 1 | 0 | 0 | 1 | 0 | 41 | 0 | 54 |
| Move Playingcard Away | 43 | 11 | 53 | 22 | 36 | 0 | 47 | 0 | 68 | 3 | 69 |
| Move Stapler Pad | 2 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 12 | 0 | 12 |
| Open Laptop | 59 | 32 | 85 | 46 | 56 | 0 | 49 | 0 | 82 | 7 | 31 |
| Open Microwave | 37 | 20 | 80 | 50 | 86 | 0 | 5 | 0 | 61 | 22 | -- |
| Pick Diverse Bottles | 2 | 0 | 27 | 6 | 7 | 0 | 6 | 0 | 52 | 1 | 30 |
| Pick Dual Bottles | 42 | 13 | 57 | 12 | 31 | 0 | 24 | 0 | 60 | 1 | 43 |
| Place A2B Left | 3 | 1 | 31 | 1 | 1 | 0 | 2 | 0 | 46 | 2 | 20 |
| Place A2B Right | 1 | 1 | 27 | 6 | 0 | 0 | 13 | 0 | 49 | 0 | 22 |
| Place Bread Basket | 10 | 2 | 17 | 4 | 6 | 0 | 14 | 0 | 26 | 1 | 52 |
| Place Bread Skillet | 5 | 1 | 23 | 1 | 7 | 0 | 11 | 0 | 19 | 0 | 56 |
| Place Burger Fries | 50 | 27 | 80 | 4 | 49 | 0 | 72 | 0 | 72 | 18 | 96 |
| Place Can Basket | 19 | 6 | 41 | 5 | 1 | 0 | 18 | 0 | 67 | 2 | 63 |
| Place Cans Plasticbox | 6 | 5 | 34 | 2 | 16 | 0 | 40 | 0 | 48 | 3 | 81 |
| Place Container Plate | 78 | 17 | 88 | 45 | 72 | 1 | 41 | 0 | 86 | 1 | 99 |
| Place Dual Shoes | 4 | 4 | 15 | 0 | 9 | 0 | 8 | 0 | 13 | 0 | 28 |
| Place Empty Cup | 56 | 7 | 37 | 11 | 61 | 0 | 37 | 0 | 65 | 1 | 72 |
| Place Fan | 12 | 2 | 20 | 10 | 1 | 0 | 3 | 0 | 36 | 1 | 28 |
| Place Mouse Pad | 1 | 0 | 7 | 1 | 0 | 0 | 0 | 0 | 4 | 1 | 9 |
| Place Object Basket | 33 | 17 | 16 | 2 | 15 | 0 | 15 | 0 | 65 | 0 | 40 |
| Place Object Scale | 1 | 0 | 10 | 0 | 0 | 0 | 1 | 0 | 15 | 0 | 19 |
| Place Object Stand | 15 | 5 | 36 | 11 | 1 | 0 | 22 | 0 | 60 | 0 | 48 |
| Place Phone Stand | 15 | 6 | 35 | 7 | 2 | 0 | 13 | 0 | 44 | 2 | 24 |
| Place Shoe | 35 | 7 | 28 | 6 | 5 | 0 | 23 | 0 | 58 | 2 | 63 |
| Press Stapler | 41 | 24 | 62 | 29 | 31 | 6 | 6 | 0 | 69 | 3 | 60 |
| Put Bottles Dustbin | 21 | 4 | 54 | 13 | 27 | 1 | 22 | 0 | 60 | 21 | -- |
| Put Object Cabinet | 33 | 18 | 68 | 18 | 15 | 0 | 42 | 0 | 72 | 1 | 35 |
| Rotate QRcode | 50 | 5 | 68 | 15 | 1 | 0 | 13 | 0 | 74 | 1 | 50 |
| Scan Object | 4 | 1 | 18 | 1 | 2 | 0 | 9 | 0 | 31 | 1 | 13 |
| Shake Bottle Horizontally | 84 | 51 | 99 | 51 | 63 | 4 | 59 | 18 | 100 | 25 | 98 |
| Shake Bottle | 74 | 45 | 97 | 60 | 74 | 10 | 65 | 8 | 98 | 19 | 98 |
| Stack Blocks Three | 2 | 0 | 17 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 41 |
| Stack Blocks Two | 21 | 2 | 42 | 1 | 25 | 0 | 7 | 0 | 24 | 0 | 83 |
| Stack Bowls Three | 51 | 17 | 66 | 24 | 48 | 0 | 63 | 0 | 57 | 5 | 62 |
| Stack Bowls Two | 76 | 30 | 91 | 41 | 82 | 0 | 61 | 0 | 83 | 6 | 90 |
| Stamp Seal | 1 | 0 | 3 | 4 | 2 | 0 | 2 | 0 | 18 | 0 | 27 |
| Turn Switch | 35 | 15 | 27 | 23 | 5 | 2 | 36 | 1 | 46 | 8 | 26 |
| **平均值** | **34.50** | **13.72** | **46.42** | **16.34** | **29.74** | **1.74** | **28.04** | **0.64** | **55.24** | **4.96** | **50.38** |

*注：全部 50 项任务在同一个模型中训练，每项任务使用 50 条示范（共 50×50 条示范）。模型检查点可从 [Qwen3-VL-OFT-Robotwin2](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robotwin2) 下载。*

</details>


<details open>
<summary><b>RoboTwin 2.0 扩大数据规模的设置</b></summary>

### 训练数据集

模型使用官方 **RobotWin 2.0 数据集**进行训练。

* 无随机化示范数据：50 项任务 × 每项任务 50 条轨迹
* 随机化示范数据：50 项任务 × 每项任务 500 条轨迹

### StarVLA-OFT

| 任务                      | 简单       | 困难       |
| ------------------------- | ---------- | ---------- |
| stack_blocks_two          | 1.0000     | 1.0000     |
| place_cans_plasticbox     | 1.0000     | 0.9900     |
| grab_roller               | 1.0000     | 1.0000     |
| place_empty_cup           | 1.0000     | 1.0000     |
| shake_bottle_horizontally | 1.0000     | 1.0000     |
| lift_pot                  | 1.0000     | 1.0000     |
| place_burger_fries        | 1.0000     | 1.0000     |
| move_playingcard_away     | 1.0000     | 0.9800     |
| adjust_bottle             | 1.0000     | 0.9900     |
| shake_bottle              | 1.0000     | 1.0000     |
| blocks_ranking_rgb        | 0.9900     | 0.9800     |
| stack_bowls_two           | 0.9900     | 1.0000     |
| place_container_plate     | 0.9900     | 0.9900     |
| press_stapler             | 0.9900     | 0.9600     |
| place_object_stand        | 0.9900     | 0.9800     |
| open_laptop               | 0.9800     | 1.0000     |
| handover_mic              | 0.9800     | 0.9600     |
| move_pillbottle_pad       | 0.9800     | 1.0000     |
| handover_block            | 0.9700     | 0.9300     |
| place_shoe                | 0.9600     | 1.0000     |
| stack_bowls_three         | 0.9500     | 0.9100     |
| place_fan                 | 0.9400     | 0.9500     |
| scan_object               | 0.9400     | 0.9100     |
| stack_blocks_three        | 0.9400     | 0.8600     |
| place_object_basket       | 0.9300     | 0.9400     |
| beat_block_hammer         | 0.9300     | 0.9200     |
| place_object_scale        | 0.9300     | 0.9300     |
| place_dual_shoes          | 0.9100     | 0.8900     |
| pick_dual_bottles         | 0.9100     | 0.9300     |
| place_bread_basket        | 0.9100     | 0.7800     |
| dump_bin_bigbin           | 0.9100     | 0.9400     |
| move_can_pot              | 0.9100     | 0.9000     |
| put_bottles_dustbin       | 0.9000     | 0.8500     |
| place_a2b_left            | 0.9000     | 0.9500     |
| place_bread_skillet       | 0.8900     | 0.8000     |
| put_object_cabinet        | 0.8900     | 0.9100     |
| place_a2b_right           | 0.8800     | 0.9500     |
| rotate_qrcode             | 0.8800     | 0.9000     |
| pick_diverse_bottles      | 0.8700     | 0.8600     |
| place_mouse_pad           | 0.8700     | 0.9400     |
| stamp_seal                | 0.8600     | 0.9000     |
| place_phone_stand         | 0.8600     | 0.9500     |
| blocks_ranking_size       | 0.7900     | 0.8000     |
| place_can_basket          | 0.7500     | 0.7500     |
| move_stapler_pad          | 0.7400     | 0.9000     |
| turn_switch               | 0.6500     | 0.6200     |
| click_alarmclock          | 0.5800     | 0.5100     |
| hanging_mug               | 0.3400     | 0.2900     |
| open_microwave            | 0.2800     | 0.3900     |
| click_bell                | 0.2300     | 0.2700     |
| **平均值**               | **0.8818** | **0.8832** |

*注：全部 50 项任务在同一个模型中训练，每项任务使用 50 + 500 条示范（共 50×550 条示范）。模型检查点可从 [Qwen3-VL-OFT-Robotwin2-All](https://huggingface.co/StarVLA/Qwen3-VL-OFT-RoboTwin2-All) 下载。*


</details>

---



# 评测

## 📦 1. 环境配置

请先按照 [RoboTwin 官方安装指南](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)创建基础 `robotwin-star` 环境。

然后完成以下一次性配置，准备好两个运行环境：

1. 在 `starvla` 环境中安装 StarVLA 依赖。

```bash
conda activate starvla
pip install -r requirements.txt
```

2. 在 `robotwin-star` 环境中安装 RoboTwin 评测端依赖。

```bash
conda activate robotwin-star
pip install -r examples/Robotwin/eval_files/requirements.txt
```

3. 将启动器指向本地的 RoboTwin 仓库，并指定使用 `robotwin-star` 评测环境。

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_ENV=robotwin-star
```

4. 由于 RoboTwin 是第三方仓库，需要修改你本地的 RoboTwin 仓库，使 `script/eval_policy.py` 支持 `--policy_ckpt_path` 参数。

在你本地的 RoboTwin 仓库中应用以下修改：

```diff
diff --git a/script/eval_policy.py b/script/eval_policy.py
index eded198..9fb36e3 100644
--- a/script/eval_policy.py
+++ b/script/eval_policy.py
@@ -69,6 +69,7 @@ def main(usr_args):
     # checkpoint_num = usr_args['checkpoint_num']
     policy_name = usr_args["policy_name"]
     instruction_type = usr_args["instruction_type"]
+    policy_ckpt_path = usr_args["policy_ckpt_path"]
     save_dir = None
     video_save_dir = None
     video_size = None
@@ -81,6 +82,7 @@ def main(usr_args):
     args['task_name'] = task_name
     args["task_config"] = task_config
     args["ckpt_setting"] = ckpt_setting
+    args["policy_ckpt_path"] = policy_ckpt_path

     embodiment_type = args.get("embodiment")
     embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
@@ -327,11 +329,13 @@ def eval_policy(task_name,
 def parse_args_and_config():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", type=str, required=True)
+    parser.add_argument("--policy_ckpt_path", type=str, required=True)
     parser.add_argument("--overrides", nargs=argparse.REMAINDER)
     args = parser.parse_args()

     with open(args.config, "r", encoding="utf-8") as f:
         config = yaml.safe_load(f)
+    config["policy_ckpt_path"] = args.policy_ckpt_path

     # Parse overrides
     def parse_override_pairs(pairs):
```

由于 RoboTwin 在独立仓库中维护，此处仅记录补丁内容，并未将其纳入 `starVLA` 仓库。StarVLA 启动器会在运行时传入 `--policy_ckpt_path`；如果没有应用此补丁，RoboTwin 就无法将检查点路径传递给 `model2robotwin_interface.py`。

可选配置：

- 如果需要脚本自动执行初始化所需的 `pip install` 步骤，请设置 `export ROBOTWIN_AUTO_INSTALL_DEPS=1`。
- 如果你的 conda 环境名称与默认值不同，请设置 `ROBOTWIN_STARVLA_ENV` 和 `ROBOTWIN_ENV`。

## 🚀 2. 评测流程

### 推荐方式：`start_eval.sh`（统一入口）

`start_eval.sh` 是主启动脚本。它会启动策略服务器，等待服务就绪后运行 RoboTwin 评测，在每个回合结束后向终端实时输出成功率，并在退出时（包括按下 Ctrl+C）清理所有进程。

```
bash start_eval.sh -m <mode> -n <policy_name> -c <ckpt_path> [options] <tasks...>
```

#### 必填选项

| 选项 | 说明 |
|------|-------------|
| `-m`, `--mode` | 评测模式：`demo_clean` 或 `demo_randomized` |
| `-n`, `--name` | 策略名称（用于命名日志目录，并以 `ckpt_setting` 参数传递给 RoboTwin） |
| `-c`, `--ckpt` | StarVLA 检查点文件的路径 |

#### 任务（位置参数）

选项之后的所有剩余参数都会被视为任务。你可以指定：

- 一个或多个任务名称：`adjust_bottle open_laptop lift_pot`
- 关键字 `all`，用于评测 RoboTwin 2.0 的全部 50 项任务
- 任务列表文件（每行一个任务）：`task_list.txt`

#### 可选选项

| 选项 | 默认值 | 说明 |
|------|---------|-------------|
| `-s`, `--seed` | `0` | 评测随机种子（也可通过 `ROBOTWIN_SEED` 设置） |
| `-j`, `--jobs-per-gpu` | `1` | 每张可见 GPU 上的并发作业数（也可通过 `ROBOTWIN_JOBS_PER_GPU` 设置） |
| `-p`, `--base-port` | `5694` | 分配端口时的起始端口号（也可通过 `ROBOTWIN_BASE_PORT` 设置） |
| `--server-timeout` | `600` | 等待策略服务器启动的时间，单位为秒（也可通过 `ROBOTWIN_SERVER_TIMEOUT` 设置） |
| `--install-deps` | 关闭 | 执行一次 pip install 依赖初始化步骤（也可通过 `ROBOTWIN_AUTO_INSTALL_DEPS=1` 设置） |
| `-h`, `--help` | | 显示帮助信息 |

同时设置命令行选项和环境变量时，命令行选项优先。

#### 示例

在无随机化模式下评测单项任务：

```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean -n test1 \
    -c /path/to/checkpoint.pt \
    adjust_bottle
```

评测多项任务：

```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_randomized -n my_run \
    -c /path/to/checkpoint.pt \
    adjust_bottle open_laptop lift_pot place_shoe
```

使用自定义随机种子评测全部 50 项任务，每张 GPU 并发运行 2 个作业：

```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean -n full_eval -s 42 -j 2 \
    -c /path/to/checkpoint.pt \
    all
```

从文件读取任务：

```bash
bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_clean -n my_run \
    -c /path/to/checkpoint.pt \
    task_list.txt
```

### 多 GPU 调度

启动器会通过 `CUDA_VISIBLE_DEVICES` 或 `nvidia-smi` 自动检测可见 GPU，默认在每张 GPU 上运行一组策略服务器和评测进程。端口从 `--base-port` 指定的值开始自动分配。

使用 8 张 GPU 的示例：

```bash
export ROBOTWIN_PATH=/path/to/RoboTwin
export ROBOTWIN_ENV=robotwin-star
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash examples/Robotwin/eval_files/start_eval.sh \
    -m demo_randomized -n full_eval \
    -c /path/to/checkpoint.pt \
    all
```

此配置会将全部 50 项任务调度到 8 张 GPU 上，最多同时运行 8 项任务。当某张 GPU 上的任务完成后，下一个待执行任务会被分配到该空闲槽位。

### 运行时输出

评测期间，每个回合结束后都会向标准输出实时报告成功率：

```
[RESULT] adjust_bottle: Success rate: 1/1 => 100.0%, current seed: 100001
[RESULT] adjust_bottle: Success rate: 2/2 => 100.0%, current seed: 100002
[RESULT] adjust_bottle: Success rate: 3/3 => 100.0%, current seed: 100005
```

完整的评测输出（包括每一步的日志）始终会保存到日志文件中。

### 进程清理

按下 Ctrl+C（或发送 SIGINT/SIGTERM）会触发递归清理，终止整个进程树，包括所有策略服务器和 RoboTwin 评测子进程。清理时会先发送 SIGTERM，2 秒后再向仍未退出的进程发送 SIGKILL。

### 日志

日志默认写入检查点所在目录下：

```
<ckpt_dir>/robotwin_eval_logs/<name>_<mode>_<ckpt_stem>_<timestamp>/
    <task>_<mode>_slot<N>_gpu<G>_port<P>_server.log
    <task>_<mode>_slot<N>_gpu<G>_port<P>_eval.log
```

可通过 `ROBOTWIN_LOG_ROOT` 覆盖默认的日志根目录。

### 环境变量

未设置对应的命令行选项时，启动器会读取以下环境变量：

| 变量 | 默认值 / 本文配置值 | 说明 |
|----------|---------|-------------|
| `ROBOTWIN_PATH` | — | 本地 RoboTwin 仓库的路径（必填） |
| `ROBOTWIN_STARVLA_ENV` | `starvla` | 策略服务器使用的 conda 环境名称（用于自动检测 Python） |
| `ROBOTWIN_ENV` | `robotwin-star`（按上文设置） | RoboTwin 评测使用的 conda 环境名称（用于自动检测 Python） |
| `STARVLA_PYTHON` | 自动 | 显式指定 starvla 环境中的 Python 可执行文件路径（跳过 conda 环境查找） |
| `ROBOTWIN_PYTHON` | 自动 | 显式指定 robotwin-star 环境中的 Python 可执行文件路径（跳过 conda 环境查找） |
| `ROBOTWIN_SEED` | `0` | 评测随机种子（可由 `-s` 覆盖） |
| `ROBOTWIN_JOBS_PER_GPU` | `1` | 每张 GPU 上的并发作业数（可由 `-j` 覆盖） |
| `ROBOTWIN_BASE_PORT` | `5694` | 分配端口时的起始端口号（可由 `-p` 覆盖） |
| `ROBOTWIN_SERVER_TIMEOUT` | `600` | 服务器启动超时时间，单位为秒（可由 `--server-timeout` 覆盖） |
| `ROBOTWIN_AUTO_INSTALL_DEPS` | `0` | 设为 `1` 时使用 pip 初始化依赖（可由 `--install-deps` 覆盖） |
| `ROBOTWIN_LOG_ROOT` | 自动 | 覆盖默认的日志输出目录 |

启动器**不会**使用 `conda activate`，而是直接从 conda 环境目录中定位 Python 可执行文件。它会搜索 `CONDA_EXE`、`CONDA_PREFIX`、`~/miniconda3/envs/`、`~/anaconda3/envs/` 等位置。如果自动检测失败，请显式设置 `STARVLA_PYTHON` 和 `ROBOTWIN_PYTHON`。

### `deploy_policy.yml` 配置

`examples/Robotwin/eval_files/deploy_policy.yml` 用作配置模板。运行时会从中读取以下字段：

| 字段 | 说明 |
|-------|-------------|
| `normalization_mode` | 归一化模式：`min_max` 或 `q99` |
| `unnorm_key` | 机器人本体对应的反归一化键 |
| `action_mode` | 动作模式（例如 `abs`） |

启动器会在运行时覆盖 `host` 和 `port`。如果你的检查点在训练时使用了分位数归一化，请设置 `normalization_mode: "q99"`。

### 底层手动模式

如果你希望自行管理策略服务器和评测进程，可以按以下步骤操作：

1. 启动策略服务器（在 `starvla` conda 环境中）：

```bash
bash examples/Robotwin/eval_files/run_policy_server.sh /path/to/checkpoint.pt [gpu_id] [port]
```

2. 运行评测（在 `robotwin-star` conda 环境中）：

```bash
conda activate robotwin-star
cd examples/Robotwin/eval_files
bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id> <ckpt_path> [port] [host]
```

示例：

```bash
bash eval.sh adjust_bottle demo_clean my_eval 0 0 /path/to/checkpoint.pt 5694
```

### RoboTwin 2.0 任务列表

RoboTwin 2.0 的全部任务如下：

```txt
adjust_bottle
beat_block_hammer
blocks_ranking_rgb
blocks_ranking_size
click_alarmclock
click_bell
dump_bin_bigbin
grab_roller
handover_block
handover_mic
hanging_mug
lift_pot
move_can_pot
move_pillbottle_pad
move_playingcard_away
move_stapler_pad
open_laptop
open_microwave
pick_diverse_bottles
pick_dual_bottles
place_a2b_left
place_a2b_right
place_bread_basket
place_bread_skillet
place_burger_fries
place_can_basket
place_cans_plasticbox
place_container_plate
place_dual_shoes
place_empty_cup
place_fan
place_mouse_pad
place_object_basket
place_object_scale
place_object_stand
place_phone_stand
place_shoe
press_stapler
put_bottles_dustbin
put_object_cabinet
rotate_qrcode
scan_object
shake_bottle_horizontally
shake_bottle
stack_blocks_three
stack_blocks_two
stack_bowls_three
stack_bowls_two
stamp_seal
turn_switch
```

支持的模式包括 `demo_clean` 和 `demo_randomized`。
