# SenseVoiceSmall 中文域微调：过程与发现（2026-03-04）

本文记录在 `/home/zsc/Downloads/SenseVoice` 里对 `SenseVoiceSmall` 做中文域数据微调与评测期间的关键发现，方便复现与后续迭代。

## 1. 模型与代码加载

- 基座模型：`iic/SenseVoiceSmall`（ModelScope）
- 本地模型目录：`modelscope_models/iic/SenseVoiceSmall/`
- Remote code：使用仓库内 `model.py`（避免依赖远端仓库里的 python 包布局）

背景：直接从 hub 下载权重后，`trust_remote_code=True` 可能遇到 `No module named 'model'`（远端代码布局/模块名不匹配）。当前做法是把模型文件落到本地目录，并显式传入 `remote_code=model.py`。

## 2. 数据准备（wav/lab -> jsonl）

数据源（本地）：
- `/mnt/sda2/中文 - Chinese`（目录下包含配对的 `*.wav` + `*.lab`）

准备脚本：
- `data/prepare_sensevoice_jsonl_from_wav_lab.py`

输出：
- `data/train_chinese.jsonl`：`68901` 条
- `data/val_chinese.jsonl`：`1406` 条

切分策略（重要）：
- **utterance 级别随机切分**：`seed=42`，`val_ratio=0.02`
- **不是 speaker-disjoint**，因此 `val` 仍然是明显的 in-domain 验证集（可能与训练集共享说话人/录音条件/文本风格）

注：为了避免把本地数据清单提交进仓库，已在 `.gitignore` 里忽略 `data/train_chinese.jsonl`、`data/val_chinese.jsonl`。

## 3. 训练配置与损失函数

训练入口：
- `finetune.sh`

当前训练不是 LoRA-only：
- 脚本没有启用任何 `peft/lora` 的冻结逻辑，属于**常规全量参数微调**（从 `model.pt` 初始化继续训练）
- `model.py` 虽然在部分模块签名里出现 `lora_*` 参数，但仓库内没有实际注入 LoRA 层的实现

损失函数形式（代码直读）：
- 总损失：`loss = loss_ctc + loss_rich`（`model.py` 的 forward 里直接相加）
- `loss_ctc`：标准 CTC loss（主干 ASR）
- `loss_rich`：对前 4 个 “rich tokens”（语言/风格/事件/情绪等标签位）的 CE / LabelSmoothingLoss

训练日志表现：
- `loss_ctc` 数值通常远大于 `loss_rich`（`loss_rich` 常见为 `0.x~1.x`，`loss_ctc` 常为 `10~100+`），因此从标量规模看，优化主要被 CTC 项驱动。

## 4. CER 评测口径与结果

### 4.1 评测口径

评测集：
- `data/val_chinese.jsonl`
- 这是上面随机切分得到的 `val`，**不是独立 test**，也不是外部数据集

CER 归一化（评测脚本逻辑）：
- 去除 rich tag（形如 `<|...|>`）
- 去空白
- 去 Unicode 标点（Unicode category `P*`）
- `lower()`

脚本：
- `eval_cer.py`：跑全量 CER（baseline vs finetune）
- `make_cer_compare_html.py`：在同口径下做 per-sample 统计，并生成抽样对比 HTML（可听音频）

### 4.2 结果（val / in-domain）

在 2026-03-04 的一次测评中（`make_cer_compare_html.py` 的统计口径，要求 baseline/finetune 都产出 hyp 才计入）：
- scored samples：`1392`（`val` 共 `1406`）
- baseline CER：`0.120529`
- finetuned CER：`0.060447`
- 绝对降幅：`0.060082`
- 相对降幅：`49.85%`

结论解读：
- 该提升发生在 **in-domain 的 `val`** 上，且不是 speaker-disjoint；很大概率主要来自 “域内适配”（in-distribution）而非严格意义的跨域泛化提升。
- 仍存在少量回退样本（报告中用红色标注），常见为同音字/字形混淆或标点/口语词形差异。

## 5. 可视化抽样对比（HTML + 音频）

生成脚本：
- `make_cer_compare_html.py`

产物（本地，不入 git；位于 `outputs/`，默认被 `.gitignore` 忽略）：
- `outputs/cer_compare_run3/compare.html`：抽样 30 条
- `outputs/cer_compare_run3/compare_100.html`：抽样 100 条
- `outputs/cer_compare_run3/compare_100_all_in_one.html`：**单文件版**（音频 base64 内嵌，体积约 55MB）

抽样策略：
- 从 `val` 的 per-sample delta（baseline CER - finetuned CER）里选取
- 默认混合：top-improved / top-regressed / random（三类各占一部分）

## 6. 复现指令（最小集合）

1) 生成 jsonl（示例）：

```bash
python data/prepare_sensevoice_jsonl_from_wav_lab.py \
  --in_dir '/mnt/sda2/中文 - Chinese' \
  --out_train data/train_chinese.jsonl \
  --out_val data/val_chinese.jsonl \
  --val_ratio 0.02 \
  --seed 42
```

2) 微调（示例，参数可用 env 覆盖）：

```bash
TRAIN_JSONL=data/train_chinese.jsonl \
VAL_JSONL=data/val_chinese.jsonl \
MODEL_NAME_OR_MODEL_DIR=modelscope_models/iic/SenseVoiceSmall \
REMOTE_CODE=./model.py \
OUTPUT_DIR=outputs/finetune_zh_localms_run3 \
CUDA_VISIBLE_DEVICES=0 \
bash finetune.sh
```

3) CER 对比：

```bash
python eval_cer.py \
  --val_jsonl data/val_chinese.jsonl \
  --model_dir modelscope_models/iic/SenseVoiceSmall \
  --ft_init outputs/finetune_zh_localms_run3/model.pt.best
```

4) 生成对比 HTML：

```bash
python make_cer_compare_html.py \
  --val_jsonl data/val_chinese.jsonl \
  --model_dir modelscope_models/iic/SenseVoiceSmall \
  --ft_init outputs/finetune_zh_localms_run3/model.pt.best \
  --out_html outputs/cer_compare_run3/compare_100.html \
  --num_examples 100
```

单文件（音频内嵌）：

```bash
python make_cer_compare_html.py \
  --val_jsonl data/val_chinese.jsonl \
  --model_dir modelscope_models/iic/SenseVoiceSmall \
  --ft_init outputs/finetune_zh_localms_run3/model.pt.best \
  --out_html outputs/cer_compare_run3/compare_100_all_in_one.html \
  --num_examples 100 \
  --inline_audio
```

## 7. 后续建议（如果要更严谨）

- 做一个 **speaker-disjoint** 的验证/测试切分，避免共享说话人导致的“看起来很强”的提升。
- 准备一份更贴近真实使用场景的外部 test（不同录音链路/噪声/说话人），再评估 CER 改善是否仍然成立。

