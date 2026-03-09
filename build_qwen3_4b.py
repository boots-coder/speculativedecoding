import subprocess
import sys
from pathlib import Path

from config.config import HF_TOKEN


def main() -> None:
  """
  使用 mergekit 的 passthrough 配置，将 Qwen3-8B 剪枝成 “4B” 模型。

  要求：
  - 已在当前 conda 环境中安装 mergekit：`pip install -r requirements.txt`
  - 已在 shell 中激活 /home/haoxuan/miniconda3/envs/dl
  """
  root = Path(__file__).resolve().parent
  config_path = root / "mergekit_qwen3_8b_to_4b.yml"
  out_dir = root / "models" / "qwen3-4b-passthrough"
  out_dir.parent.mkdir(parents=True, exist_ok=True)

  cmd = [
    sys.executable,
    "-m",
    "mergekit.mergekit_yaml",
    str(config_path),
    str(out_dir),
    "--hf-token",
    HF_TOKEN,
  ]

  print("Running mergekit passthrough merge...")
  print("Command:", " ".join(cmd))

  subprocess.run(cmd, check=True)

  print(f"Done. Pruned model saved to: {out_dir}")


if __name__ == "__main__":
  main()

