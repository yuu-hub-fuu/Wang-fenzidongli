# AnewSampling · Table 1 基线复现（ATLAS 单体构象系综）

本仓库按照论文 *Learning the All-Atom Equilibrium Distribution of Biomolecular Interactions at Scale*（AnewSampling）**Table 1** 的设置，复现 ATLAS 测试集上的全部 8 个对比基线：

| Table 1 列 | 实际模型 / 权重 | 官方代码（固定 commit） | 采样协议（每个靶标） |
|---|---|---|---|
| ESMFlow-MD (Full) | `esmflow_md_base_202402.pt` | [bjing2016/alphaflow](https://github.com/bjing2016/alphaflow) @ `0408d7c` | `predict.py --mode esmfold --samples 250`（`tmax 1.0, steps 10`） |
| ESMFlow-MD (Distilled) | `esmflow_md_distilled_202402.pt` | 同上 | 同上 + `--noisy_first --no_diffusion` |
| ConfDiff | **ConfDiff-OF-r3-MD**（与其 README 中 ATLAS 表格一致：0.59/0.67/0.85/2.76…） | [bytedance/ConfDiff](https://github.com/bytedance/ConfDiff) @ `9cfae1c` | ColabFold MSA → OpenFold 表征（3 recycles）→ `src/eval.py experiment=full_atom data/dataset=atlas`，250 samples |
| BioEmu | `bioemu-v1.1`（Science 论文版本） | [microsoft/bioemu](https://github.com/microsoft/bioemu) @ `8babb71` | `bioemu.sample`，默认过滤非物理样本，取前 250 个；主链 + CB |
| Str2Str | `pretrain.pth`（PDB 预训练，零样本） | [lujiarui/Str2Str](https://github.com/lujiarui/Str2Str) @ `0b690e9` | 输入 ATLAS 起始结构；10 个扩散深度 × 25 replica = 250；主链输出后按其 README 用 FASPR 补侧链 |
| MDGen | ATLAS 模型 `atlas.ckpt` | [bjing2016/mdgen](https://github.com/bjing2016/mdgen) @ `81482a4` | `sim_inference.py --num_frames 250 --num_rollouts 1 --suffix _R1`（从 R1 第 0 帧滚动 100 ns） |
| EBA | `release.pt` | [lujiarui/eba](https://github.com/lujiarui/eba) @ `c1f3145` | Protenix `--predict_only`，`N_sample 250, N_step 20, λ 1.75, η 1.25, N_cycle 4, seed 42` |
| BioMD | **BioKinema `sqrt` 检查点（代理，见下）** | [IDEA-XL/BioKinema](https://github.com/IDEA-XL/BioKinema) @ `e03508e` | `atlas_benchmark/run_reproduce.sh`：从 R1/R2/R3 第 0 帧各滚动 100 ns（1 ns/帧），合并 300 帧 |

评估使用与 Table 1 **完全相同**的 AlphaFlow/BioMD 协议（论文附录 A.1），13 项指标：

| 类别 | Table 1 指标 | 计算方式 |
|---|---|---|
| Predicting flexibility | Pairwise RMSD r ↑ | 各靶标平均成对 Cα-RMSD（预测 vs MD）跨靶标 Pearson r |
| | Global RMSF r ↑ | 所有靶标所有原子 RMSF 合并后的 Pearson r |
| | Per-target RMSF r ↑ | 每个靶标 RMSF Pearson r 的中位数 |
| Distributional accuracy | Root mean W2-dist. ↓ | 逐原子 3D 高斯 W2（式 1），RMWD 的中位数 |
| | Trans. / Var. contrib. ↓ | RMWD 的平移 / 方差分量（式 2） |
| | MD PCA W2 ↓ / Joint PCA W2 ↓ | 在 MD（或 MD+预测联合）Cα PCA 前 2 维上的经验 W2 |
| | % PC-sim > 0.5 ↑ | MD 与预测第一主成分 \|cos\| > 0.5 的靶标比例 |
| Ensemble observables | Weak / Transient contacts J ↑ | Cα 8 Å 接触：晶体中存在但 ≥10% 帧断开 / 晶体中不存在但 ≥10% 帧形成 的 Jaccard |
| | Exposed residue J ↑ | 晶体中埋藏、≥10% 帧侧链 SASA > 2 Å²（探针 2.8 Å）残基的 Jaccard |
| | Exposed MI matrix rs ↑ | 暴露指示变量互信息矩阵的 Spearman 相关 |

MD 参考：ATLAS 三条 100 ns 生产轨迹合并，随机种子 137 抽取（W2/SASA 用 1000 帧，其余与预测帧数相同），所有系综先对齐到起始全原子结构。

## 目录结构

```
atlas_bench/
  metrics/core.py, analyze.py, report.py   # AlphaFlow analyze_ensembles.py / print_analysis.py 的等价移植
  paper_table1.py                          # 论文 Table 1 数值（用于对照）
  atlas_data.py                            # ATLAS 下载/校验、各模型输入（CSV/FASTA/PDB/CIF/第 0 帧）
  ensemble_io.py                           # 各基线原生输出 → 统一 {name}.pdb 多模型系综（残基编号对齐 ATLAS）
  baselines/{esmflow,confdiff,bioemu,str2str,mdgen,eba,biomd,external}.py
  tools/                                   # 在基线自身环境中运行的小工具（仅标准库/基线自带包）
  data/splits/atlas_test.csv               # AlphaFlow 的 82 个 ATLAS 测试靶标
configs/benchmark.yaml, configs/baselines/*.yaml
scripts/envs/*.sh                          # 克隆官方仓库（固定 commit）、按官方 README 建环境、下载权重
scripts/verify_against_alphaflow.py        # 与原始 AlphaFlow 评估脚本逐项数值比对
tests/                                     # 指标、各基线输出格式 → 评估全链路、推理命令协议
```

## 使用方法

> 让 AI 编程助手（Claude Code、Codex 等）代为执行时，直接让它按 [`AGENTS.md`](AGENTS.md) 操作（`CLAUDE.md` 会自动引用它）。

```bash
# 0) 评估/调度环境（CPU 即可）
pip install -e .

# 1) 下载并校验 ATLAS 测试集（82 个靶标，每个 3×100 ns）
python -m atlas_bench data --download            # -> data/atlas/{name}/{name}.pdb, {name}_prod_R{1,2,3}_fit.xtc

# 2) 为各基线建立官方环境（需要 conda + GPU/CUDA；可单独运行某一个脚本）
bash scripts/envs/all.sh                          # 或 bash scripts/envs/confdiff.sh 等

# 3) 一键运行全部 8 个基线 + 输出 Table 1（含论文数值对照）
python -m atlas_bench all --gpus 0 1 2 3

# 单个基线、单个阶段（prepare / infer / collect / evaluate）
python -m atlas_bench run confdiff --gpus 0 1
python -m atlas_bench run mdgen --stages collect,evaluate
python -m atlas_bench run eba --dry_run            # 只打印将要执行的命令

# 直接使用作者公开的 ATLAS 样本（可精确复现论文数值）：ESMFlow-MD、EBA
python -m atlas_bench run esmflow_md_full --fetch
python -m atlas_bench run eba --fetch

# 汇总任意 out.pkl 为 Table 1（Markdown/LaTeX/TSV）
python -m atlas_bench table esmflow_md_full=runs/esmflow_md_full/out.pkl confdiff=runs/confdiff/out.pkl --format latex
```

每个基线的结果位于 `runs/<baseline>/`：`inputs/`（模型输入）、`raw/`（官方代码原始输出）、`ensembles/{name}.pdb`（统一系综）、`out.pkl`（逐靶标分析）、`manifest.json`（commit、协议、帧数）。

评估 AnewSampling 或其他模型的样本：编辑 `configs/baselines/external.yaml` 中的 `patterns`，然后
`python -m atlas_bench run external --stages collect,evaluate`，再用 `table anewsampling=runs/anewsampling/out.pkl` 与论文数值对照。

## 各基线的复现细节与注意事项

- **ESMFlow-MD**：AlphaFlow 作者在 HuggingFace 公开了论文所用的 250 样本系综（`--fetch`），可以精确复现 Table 1 数值；`infer` 模式用官方权重重新采样（随机性会带来小幅差异）。
- **ConfDiff**：Table 1 中的 ConfDiff 数值与官方 README 的 `ConfDiff-OF-r3-MD` 一行完全一致，因此使用该模型。官方仓库中 `make_openfold_repr.py` 写出的表征路径/索引格式（`{ab}/{name}.node_repr.recycle3.npy`、无表头 CSV）与 `OpenFoldReprLoader` 读取的格式（`{name}/{name}_recycle3_single_repr.npy`、含 `seqres` 表头）不一致，`atlas_bench/tools/confdiff_repr_index.py` 负责桥接，不修改官方代码。
- **BioEmu**：默认使用 `bioemu-v1.1`，保留 BioEmu 默认的非物理样本过滤，自动补采直到 ≥250 个样本（`filter_samples: false` 可关闭）。输出为主链 + CB（N、CA、C、CB、O），按 AlphaFlow 的原子匹配规则，SASA 类指标只在 CB 层面计算（Table 1 中 BioEmu 的 Exposed residue J 为 “-”）；`sidechains: hpacker` 可调用官方 `bioemu.sidechain_relax` 重建完整侧链。
- **Str2Str**：零样本方法，输入为 ATLAS 起始结构（去氢）；官方默认 10 个扩散深度（0.25–0.70）×`n_replica`，取 25 得到 250 个样本。模型只输出 N/CA/C/O，默认在 collect 阶段按 Str2Str README 推荐的 [FASPR](https://github.com/tommyhuangthu/FASPR)（@ `0d55732`）逐帧补侧链（`atlas_bench/sidechain_pack.py`，保持帧顺序），因此 Exposed residue J / MI 两行也能算出；未补侧链的系综保存在 `runs/str2str/ensembles_backbone/`。设 `sidechains: none` 可回到纯主链评估。
- **MDGen**：按 MDGen README 的 ATLAS 推理命令，以 R1 第 0 帧为条件滚动 250 帧（400 ps/帧 = 100 ns）。官方 `scripts/prep_sims.py --atlas` 分支引用了未定义的 `args.atlas_dir`，因此用 `tools/mdgen_prep_atlas.py` 生成等价的 atom14 输入。
- **EBA**：作者公开了 ATLAS 测试集 250 个样本（`--fetch`）。`infer` 模式按 `sh/sample_demo.sh` 的超参数运行，并自动生成 Protenix 所需的 mmCIF/bioassembly/索引（`indices_test.csv`），同时改写 `configs/configs_data.py` 中写死的数据路径。需要 Protenix `release_data`（CCD、MSA）。
- **BioMD**：截至目前 BioMD（ICLR 2026）没有公开代码与权重。同一作者团队公开了其后续工作 **BioKinema**（论文参考文献 [24]），同样基于 Protenix 的“预测 + 插值”分层轨迹生成框架，并附带针对该 ATLAS 基准的完整复现包，因此这里以 BioKinema `sqrt` 检查点作为 BioMD 的公开代理（BioKinema 自带参考结果：RMWD 2.25 / PC-sim 45.7% / Pairwise RMSD r 0.80，与 Table 1 中 BioMD 的 2.18 / 46% / 0.70 接近但不相同）。其协议评估 81 个靶标（排除 `7aex_A`）、每个靶标 300 帧，均在 `configs/baselines/biomd.yaml` 中体现；若获得 BioMD 检查点，替换 `options.ckpt` 即可。

## 验证

- **指标移植的数值等价性**：在由 ATLAS 真实起始结构构建的合成靶标上，与原始 AlphaFlow `analyze_ensembles.py` / `print_analysis.py` 逐项比较，逐靶标最大绝对差 2.4×10⁻⁷（float32 舍入），汇总表完全一致。可在真实数据上复查：
  `python scripts/verify_against_alphaflow.py --alphaflow_repo third_party/alphaflow --atlas_dir data/atlas --pdbdir runs/esmflow_md_full/ensembles --pdb_id 6o2v_A 7ead_A`
- **测试**：`pytest`（39 项，其中 FASPR 测试在找到 FASPR 可执行文件时运行）覆盖指标性质（高斯 W2 闭式解、sqrtm、经验 W2、互信息）、8 个基线原生输出格式 → 统一系综 → 评估 → Table 1 的全链路，以及各基线推理命令中的协议参数。
- 本仓库在无 GPU 的环境中开发，**尚未在 GPU 上实际运行各基线推理**；推理命令依据各官方仓库固定 commit 的源码与 README 编写，首次运行时请检查 `runs/<baseline>/logs/`。

## 与论文数值可能存在差异的原因

1. Table 1 的基线数值来自 BioMD 论文（其又转引自各原始论文），并非统一重跑；采样随机性、ColabFold MSA 服务器随时间更新都会带来小幅差异。
2. 系综大小影响部分指标（AlphaFlow README 明确提醒）：默认均为 250 帧，BioMD/BioKinema 为 300 帧，BioEmu 过滤后若不足 250 帧会在 `manifest.json` 中记录。
3. BioMD 使用公开代理 BioKinema（见上）。
4. 为了让 Table 1 每一格都有值：Str2Str 默认用 FASPR 补侧链后再评估；BioEmu 默认用其原生输出（带 CB），SASA 类指标在 CB 层面计算（也可设 `sidechains: faspr` 或 `hpacker` 补全侧链）。论文中这两者的部分格子为 “-”，重跑后会比论文多出数值。`out.pkl` 中的 `has_cb` / `has_sidechains` 与汇总表的 “Side-chain coverage %” 记录了实际情况。
5. `table` 命令最后会检查完整性：8 个基线 × 13 项指标有任何缺列或空格都会列出，加 `--require_complete` 时以非零状态退出。AnewSampling 列需要提供其样本（`run external`），否则只显示论文数值。
