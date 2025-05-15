#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = "/home/dvcuser/dvcdata/repo"
DEST_DIR = "/home/dvcuser/dvcdata/deployed_model"

def run(cmd, cwd=None):
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)

def checkout_tag(tag_name):
    run(f"git checkout {tag_name}", cwd=REPO_DIR)
    run("dvc checkout", cwd=REPO_DIR)

def copy_model_files():
    os.makedirs(DEST_DIR, exist_ok=True)
    for file in os.listdir(REPO_DIR):
        full_file_name = os.path.join(REPO_DIR, file)
        if os.path.isfile(full_file_name) and not file.endswith(".dvc") and not file.startswith("."):
            shutil.copy(full_file_name, DEST_DIR)
            print(f"✅ Copied: {file}")

def main():
    if len(sys.argv) != 2:
        print("Usage: pull_model.py <tag_name>")
        sys.exit(1)

    tag_name = sys.argv[1]

    checkout_tag(tag_name)
    copy_model_files()
    print(f"\n✅ Model from tag '{tag_name}' copied to {DEST_DIR}")

if __name__ == "__main__":
    main()
