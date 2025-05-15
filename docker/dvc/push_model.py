#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path
import glob

SOURCE_DIR = "/home/dvcuser/dvcdata/model/distilgpt2"
REPO_DIR = "/home/dvcuser/dvcdata/repo"

def run(cmd, cwd=None):
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)

def configure_git():
    run('git config --global user.name "DVC Bot"')
    run('git config --global user.email "dvc-bot@example.com"')
    run(f"git config --global --add safe.directory {SOURCE_DIR}")
    run(f"git config --global --add safe.directory {REPO_DIR}")

def init_repo():
    os.makedirs(REPO_DIR, exist_ok=True)

    if not Path(os.path.join(REPO_DIR, ".git")).exists():
        run("git init", cwd=REPO_DIR)
        run("dvc init --no-scm", cwd=REPO_DIR)
        run("git add .", cwd=REPO_DIR)
        run("git commit -m 'Initialize local Git+DVC repo'", cwd=REPO_DIR)

def copy_models():
    for file in os.listdir(SOURCE_DIR):
        src = os.path.join(SOURCE_DIR, file)
        dst = os.path.join(REPO_DIR, file)
        if os.path.isfile(src):
            shutil.copy(src, dst)

def clean_old_dvc_files():
    for file in glob.glob(os.path.join(REPO_DIR, "*.dvc")):
        os.remove(file)

def add_and_commit_models():
    clean_old_dvc_files()
    run("dvc add *", cwd=REPO_DIR)
    run("git add .", cwd=REPO_DIR)
    run("git commit -m 'Add new model version'", cwd=REPO_DIR)

def tag_and_push(tag_name):
    run(f"git tag {tag_name}", cwd=REPO_DIR)

def main():
    if len(sys.argv) != 2:
        print("Usage: push_model.py <tag_name>")
        sys.exit(1)

    tag_name = sys.argv[1]

    configure_git()
    init_repo()
    copy_models()
    add_and_commit_models()
    tag_and_push(tag_name)

    print(f"✅ Model pushed and tagged with '{tag_name}'")

if __name__ == "__main__":
    main()
