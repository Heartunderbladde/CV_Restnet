"""
主运行脚本
按顺序执行实验全流程:
  1. 获取数据 + 标签
  2. 渲染 K 线图
  3. 训练模型
  4. 测试评估

用法:
  python run_all.py            # 从头运行全部
  python run_all.py --skip 1   # 跳过数据获取（如果已有数据）
  python run_all.py --only 3   # 只运行训练
"""

import os
import sys
import argparse
import subprocess

PYTHON = sys.executable
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = {
    "1": ("01_fetch_data.py", "获取数据 + 标签生成"),
    "2": ("02_render_charts.py", "渲染 K 线图"),
    "3": ("train.py", "训练模型"),
    "4": ("evaluate.py", "测试集评估"),
}


def run_script(filename):
    """运行一个 Python 脚本"""
    path = os.path.join(SRC_DIR, filename)
    print(f"\n{'#' * 60}")
    print(f"# 运行: {filename}")
    print(f"{'#' * 60}\n")
    result = subprocess.run([PYTHON, path], cwd=SRC_DIR)
    if result.returncode != 0:
        print(f"\nERROR: {filename} 退出码 {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="K线图涨跌分类实验")
    parser.add_argument("--skip", type=str, default="", help="跳过的步骤 (如 --skip 1,2)")
    parser.add_argument("--only", type=str, default="", help="只运行的步骤 (如 --only 3,4)")
    args = parser.parse_args()

    skip_steps = set(s.strip() for s in args.skip.split(",") if s.strip())
    only_steps = set(s.strip() for s in args.only.split(",") if s.strip())

    if only_steps:
        steps = sorted(only_steps)
    else:
        steps = [k for k in SCRIPTS if k not in skip_steps]
        steps.sort()

    print("=" * 60)
    print("基于 ResNet 的 K 线图涨跌趋势分类研究")
    print("实验全流程")
    print("=" * 60)
    print(f"\n执行步骤: {', '.join(steps)}")
    for s in steps:
        print(f"  [{s}] {SCRIPTS[s][1]}")

    for step in steps:
        filename, desc = SCRIPTS[step]
        run_script(filename)

    print(f"\n{'=' * 60}")
    print("全部完成!")
    print(f"结果目录: {os.path.join(os.path.dirname(SRC_DIR), 'outputs')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
