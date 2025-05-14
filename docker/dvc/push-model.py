#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE_DIR = "/mnt/output/model/distilgpt2"
REPO_DIR = "/home/dvcuser/dvcdata"

def run(cmd, cwd=None):
    print(f"> {cmd}")
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)

def init_repo():
    if not Path(os.path.join(REPO_DIR, ".git")).exists():
        run("git init", cwd=REPO_DIR)
        run("dvc init --no-scm", cwd=REPO_DIR)  # fallback if no git
        run("git add .dvc .gitignore", cwd=REPO_DIR)
        run("git commit -m 'init repo'", cwd=REPO_DIR)
    if not Path(os.path.join(REPO_DIR, ".dvc")).exists():
        run("dvc init", cwd=REPO_DIR)

def copy_models():
    for file in os.listdir(SOURCE_DIR):
        full_file_name = os.path.join(SOURCE_DIR, file)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, REPO_DIR)

def add_and_commit_models():
    run("dvc add *", cwd=REPO_DIR)
    run("git add .", cwd=REPO_DIR)
    run("git commit -m 'Add new model version'", cwd=REPO_DIR)

def tag_and_push(tag_name):
    run(f"git tag {tag_name}", cwd=REPO_DIR)
    run("git push origin --tags", cwd=REPO_DIR)
    run("dvc push", cwd=REPO_DIR)

def main():
    if len(sys.argv) != 2:
        print("Usage: push_model.py <tag_name>")
        sys.exit(1)

    tag_name = sys.argv[1]

    init_repo()
    copy_models()
    add_and_commit_models()
    tag_and_push(tag_name)
    print(f"✅ Model pushed and tagged with '{tag_name}'")

if __name__ == "__main__":
    main()
