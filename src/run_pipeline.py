#!/usr/bin/env python
"""一键运行：数据处理 → 特征工程 → 模型训练"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    ("[1/3] 数据管道", "src/data_pipeline.py"),
    ("[2/3] 特征工程", "src/features.py"),
    ("[3/3] 模型训练", "src/models.py"),
]

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def main():
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}  VCT 全流程流水线{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")

    for name, script in SCRIPTS:
        path = Path(script)
        if not path.exists():
            print(f"  {RED}[错误] 找不到 {script}{RESET}")
            sys.exit(1)

        print(f"{BOLD}{name}{RESET} 正在运行 {CYAN}{script}{RESET} ...\n")
        result = subprocess.run([sys.executable, script])

        if result.returncode != 0:
            print(f"\n  {RED}[失败] [{name}] 退出码 {result.returncode}{RESET}")
            sys.exit(1)
        print(f"\n  {GREEN}[完成] [{name}]{RESET}\n")

    print(f"{GREEN}{'='*60}{RESET}")
    print(f"{GREEN}  全部完成！运行以下命令启动应用：{RESET}")
    print(f"{GREEN}  streamlit run app/main.py{RESET}")
    print(f"{GREEN}{'='*60}{RESET}")


if __name__ == "__main__":
    main()
