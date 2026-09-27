# AI 执行手册：在 GPU 服务器上复现 Table 1 基线

> 给 Claude Code / Codex / Cursor 等编程助手看的操作说明。人类读者请先看 `README.md`。
> 目标：在 ATLAS 测试集上跑出 AnewSampling 论文 Table 1 的 8 个基线，输出与论文数值对照的表格。

## 0. 规则（必须遵守）

- **不要修改评估代码** `atlas_bench/metrics/` 和 `atlas_bench/paper_table1.py`。它们与 AlphaFlow 官方评估脚本逐项等价，改动会让结果与论文不可比。
- **不要改采样协议参数**（样本数 250、各基线的步数/温度/种子等，见 `configs/baselines/*.yaml` 与 `README.md`）。确需改动（如显存不足需要调小 batch）时，只改不影响分布的参数（batch 大小、GPU 数），并在最终报告中写明。
- **不要修改 `third_party/` 下的官方仓库源码**。例外：EBA 的 `configs/configs_data.py` 数据路径由本框架自动改写，属预期行为。
- 每一步结束后按"检查点"验证，失败先看日志 `runs/<baseline>/logs/*.log`，再按第 6 节排查。
- 单个基线反复失败（同一问题尝试修复 3 次以上）时，记录原因后跳过，继续其他基线，最后在报告中说明。
- 长时间任务放在 `tmux`/`nohup` 中运行，不要让终端会话断开导致任务中止。

## 1. 环境检查

```bash
nvidia-smi                       # 需要 NVIDIA GPU；建议 A100/H100 80GB，≥1 张
nvcc --version || true           # AlphaFlow/ConfDiff 的 OpenFold 需要 CUDA 11.x 工具链（可装在 conda 环境内）
conda --version                  # 需要 conda 或 mamba
df -h .                          # 预留 ≥300 GB：ATLAS 数据 + 8 套环境/权重 + 生成的系综
python3 --version                # ≥3.9，用于本框架（评估只用 CPU）
```

网络需能访问：`github.com`、`huggingface.co`、`www.dsimb.inserm.fr`（ATLAS）、`storage.googleapis.com`（EBA）、`af3-dev.tos-cn-beijing.volces.com`（Protenix 数据）、`drive.google.com`（Str2Str 权重）、`api.colabfold.com`（MSA 服务器，ConfDiff/BioEmu/BioKinema 会用）、`download.pytorch.org`、`dl.fbaipublicfiles.com`（ESM 权重）。

## 2. 安装本框架并自检

```bash
pip install -e .
pytest -q                        # 检查点：37 passed
python -m atlas_bench list       # 列出 8 个基线及其官方仓库/commit
```

## 3. 下载 ATLAS 测试集

```bash
python -m atlas_bench data --download --workers 8
```

检查点：最后一行为 `82/82 ATLAS targets complete in .../data/atlas`。缺失的靶标重跑同一命令即可（会跳过已完成的）。
每个靶标应有 `data/atlas/{name}/{name}.pdb` 和 `{name}_prod_R{1,2,3}_fit.xtc`。**必须是 `_fit.xtc`**（已处理周期性边界），不要用未拟合的原始轨迹代替。

## 4. 建立各基线环境

```bash
bash scripts/envs/all.sh 2>&1 | tee setup.log      # 或逐个：bash scripts/envs/<name>.sh
```

脚本会把官方仓库克隆到 `third_party/`（固定 commit）、建 conda 环境、下载权重。环境名与 `configs/baselines/*.yaml` 中的 `python:` 一致：

| 基线 | 脚本 | conda 环境 | 需要确认的文件 |
|---|---|---|---|
| esmflow_md_full / distilled | `alphaflow.sh` | `alphaflow` | `third_party/alphaflow/params/esmflow_md_{base,distilled}_202402.pt` |
| confdiff | `confdiff.sh` | `confdiff` | `third_party/ConfDiff/checkpoints/ConfDiff-MD/` 下文件名含 `OF-r3` 的 ckpt；OpenFold 参数 `pretrain_repr/openfold/openfold_params/finetuning_no_templ_ptm_1.pt` |
| bioemu | `bioemu.sh` | `bioemu` | 无（首次运行自动下载） |
| str2str | `str2str.sh` | `str2str` | `third_party/Str2Str/data/ckpt/pretrain.pth` |
| mdgen | `mdgen.sh` | `mdgen` | `third_party/mdgen/weights/` 下的 ATLAS 模型 |
| eba | `eba.sh` | `eba` | `third_party/eba/release.pt`、`third_party/eba/cutlass/`、`third_party/protenix_data/seq_to_pdb_index.json` |
| biomd | `biomd_biokinema.sh` | `biokinema` | `third_party/BioKinema/checkpoints/BioKinema_atlas+misato+mdposit_sqrt.pt` |

检查点：每个脚本以 `[setup] ... ready` 结尾。**权重文件名与上表不一致时**（尤其 ConfDiff、MDGen，官方页面未写明文件名），用 `ls` 找到实际文件，把路径写入对应 `configs/baselines/<name>.yaml` 的 `options.ckpt`。

## 5. 运行

### 5.1 冒烟测试（先做，约 1 小时）

用最短的两个靶标把每个基线完整跑一遍：

```bash
for b in esmflow_md_full esmflow_md_distilled confdiff bioemu str2str mdgen eba biomd; do
  python -m atlas_bench run $b --targets 7lp1_A 6ro6_A --gpus 0 2>&1 | tee smoke_$b.log
done
```

检查点（每个基线）：
- 输出 `collected 2/2 ensembles`，`runs/<b>/ensembles/` 下有 `7lp1_A.pdb`、`6ro6_A.pdb`；
- `runs/<b>/manifest.json` 中每个靶标的帧数为 250（biomd 为 300；bioemu 过滤后仍不足时会少于 250，见第 7 节）；
- 出现 `Analyzed 2/2 targets`，生成 `runs/<b>/out.pkl`。

只看命令不执行：加 `--dry_run`。只重跑某几个阶段：`--stages collect,evaluate`。

冒烟测试通过后，**删除冒烟产物**再跑全量，避免混入：`rm -rf runs/`（第 3、4 步的数据与环境不受影响）。

### 5.2 全量运行

GPU 小时粗估（单张 A100 80GB）：ESMFlow-MD Full 40–60、Distilled ~5、ConfDiff 5–10、BioEmu ~22、Str2Str 3–5、MDGen 1–3、EBA 30–40、BioMD 10–20。

推荐先用作者公开的样本（不需要 GPU，且能精确对应论文数值）：

```bash
python -m atlas_bench run esmflow_md_full --fetch
python -m atlas_bench run esmflow_md_distilled --fetch
python -m atlas_bench run eba --fetch
```

其余基线用官方代码推理（把 `--gpus` 换成实际可用的卡号）：

```bash
nohup python -m atlas_bench all --baselines confdiff bioemu str2str mdgen biomd --gpus 0 1 2 3 \
  > run_all.log 2>&1 &
```

时间充裕时，也可以对 ESMFlow-MD 和 EBA 再跑一次官方推理（去掉 `--fetch`，另设 `run_name` 以免覆盖），作为复现一致性检查。

检查点：`run_all.log` 中每个基线 `collected 82/82`（biomd 为 81/81，按其官方协议排除 `7aex_A`），以及 `Analyzed N/N targets`。

## 6. 汇总结果

```bash
python -m atlas_bench table \
  esmflow_md_full=runs/esmflow_md_full/out.pkl esmflow_md_distilled=runs/esmflow_md_distilled/out.pkl \
  confdiff=runs/confdiff/out.pkl bioemu=runs/bioemu/out.pkl str2str=runs/str2str/out.pkl \
  mdgen=runs/mdgen/out.pkl eba=runs/eba/out.pkl biomd=runs/biomd/out.pkl \
  --format markdown --csv results_summary.csv | tee results_table.md
```

`[paper]` 列为论文数值。合理性判断：
- `--fetch` 得到的 ESMFlow-MD、EBA 应与论文数值非常接近（通常差 ≤0.02，PC-sim 差几个百分点）；差距明显时优先排查评估步骤（ATLAS 数据是否完整、是否用了 `_fit.xtc`）。
- BioMD 用的是 BioKinema 代理，参考值为 BioKinema 自带的 `third_party/BioKinema/experiments/atlas_benchmark/expected_metrics.txt`（RMWD 2.25、PC-sim 45.7%、Pairwise RMSD r 0.80），与论文中 BioMD 的数值本来就不完全相同。
- Str2Str 的 Exposed residue J / MI 为 NaN 属正常（只有主链）；BioEmu 的 SASA 类指标只在 CB 层面计算。

最后写 `RESULTS.md`，包括：结果表、每个基线用的是 fetch 还是推理、实际帧数、失败或跳过的靶标及原因、任何偏离默认配置的改动、硬件与总耗时。

## 7. 常见问题排查

| 现象 | 处理 |
|---|---|
| OpenFold 安装失败（alphaflow/confdiff） | 需要 CUDA 11：在该 conda 环境内 `conda install nvidia/label/cuda-11.8.0::cuda` 等（见 AlphaFlow README 的 Installation 一节），然后 `CUDA_HOME=$CONDA_PREFIX pip install ...openfold...` |
| ConfDiff 报 `node_repr not found` / 找不到 `seqres_to_index.recycle3.csv` | 确认表征阶段成功，并已运行 `atlas_bench/tools/confdiff_repr_index.py`（框架自动执行）；可用 `--stages infer` 重跑，已生成的表征会跳过 |
| ConfDiff 自动找错 ckpt | 在 `configs/baselines/confdiff.yaml` 中显式设置 `options.ckpt` 为 OF-r3-MD 的 ckpt |
| MDGen 找不到 `weights/atlas.ckpt` | `ls third_party/mdgen/weights/`，把 ATLAS 模型路径写入 `configs/baselines/mdgen.yaml` 的 `options.ckpt` |
| EBA 数据加载报错 / 缺 MSA | 确认 `third_party/protenix_data/` 已解压 release_data；ATLAS 序列不在其 MSA 索引中时，按 Protenix 的方式用 `third_party/eba/runner/msa_search.py` 生成。**或直接用 `--fetch`** |
| BioKinema 内核编译失败 | 在 `configs/baselines/biomd.yaml` 设置 `options.cutlass_path`（CUTLASS v3.5.1）与 `options.cuda_home`（CUDA 11.8），确保 `ninja` 在 PATH 中 |
| BioEmu 样本不足 250 | 框架会自动补采最多 4 轮；仍不足时调大 `options.oversample` 或设 `filter_samples: false`，并在报告中说明 |
| 某靶标 CUDA OOM | 减小该基线的 batch 参数（ConfDiff `gen_batch_size`、Str2Str `replica_per_batch`、BioEmu `batch_size_100`），不改样本数 |
| `collect` 报序列不匹配 | 查看报错中打印的两条序列；通常是输入用错了靶标，不要放宽匹配条件 |
| 评估报 `No common atoms` | 预测结构的残基/原子命名异常；检查 `runs/<b>/ensembles/{name}.pdb` 的前几行 |

需要确认评估代码与 AlphaFlow 原版一致时：
`python scripts/verify_against_alphaflow.py --alphaflow_repo third_party/alphaflow --atlas_dir data/atlas --pdbdir runs/esmflow_md_full/ensembles --pdb_id 6o2v_A 7ead_A`（最后一行的差值应 < 1e-4）。
