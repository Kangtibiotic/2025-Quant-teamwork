# README

## 项目简介

本仓库为量化交易课程大作业代码与数据处理流水线，包含数据预处理、因子生成、因子处理、单因子检验、多因子建模与样本外回测等脚本。

## 数据下载
链接: https://pan.sjtu.edu.cn/web/share/b5ff5c2d5f7412aaf02a65d4cb71936a, 提取码: tpot

## 目录结构（简要）

- `raw_data/`：原始源数据（请从交大云盘下载）
- `processed_data/`：预处理后产生的中间数据（请从交大云盘下载）
- `raw_factors/`：单因子原始输出（请从交大云盘下载）
- `scripts/`：所有可执行的脚本，按序编号命名
- `results/`：回测与分析结果

（详见仓库实际目录）

## 快速开始

1. 克隆仓库并切到项目根目录（本 `README.md` 所在目录）。
2. 创建并激活 Python 环境，建议 Python 3.9+：

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\\Scripts\\activate     # Windows PowerShell
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

4. 获取原始数据：将交大云盘中的 `raw_data/`、`raw_factors/`复制到仓库根目录。若从头运行，请确保 `raw_data/` 可用。

5. 按序执行脚本（从 0 到 7），示例：

```bash
python scripts/0.数据预处理.py
# 运行后，需要将raw_data的其他文件拷贝至processed_data以继续。
python scripts/1.因子生成/Amihud.py
python scripts/1.因子生成/mom_1m.py
...  # 依次运行其他脚本
python scripts/7.绩效分析.py
```

建议先运行少量脚本来确认环境正确再全量运行。`trend_strength` 因子生成较慢，约需 20–40 分钟，根据机器差异而定。

## 运行流程要点

- 请在项目根目录运行脚本（即包含 `scripts/` 的目录）。
- 所有脚本均依赖前一步的输出，请按照序号顺序运行。

## 环境依赖

- Python 3.9+
- 安装依赖：

```bash
pip install -r requirements.txt
```

## 常见问题与排查建议

- 如果报找不到文件（FileNotFoundError）：确认 `raw_data/` 与 `processed_data/` 的目录结构未被移动或重命名。
- 如果报模块导入错误：确认已在项目根目录并使用了正确的 Python 环境，使用 `pip list` 检查依赖是否安装。
- 如果单因子生成中断或超时：先运行单个因子脚本以复现问题，再检查日志或在脚本中添加打印以定位耗时环节。