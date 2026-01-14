"""
Generate standalone Kaggle notebook with all NanoPrime code embedded
"""
import json

# Read source files
files_to_embed = {
    'src/config.py': open('src/config.py', 'r', encoding='utf-8').read(),
    'src/__init__.py': '',
    'data/cosmopedia.py': open('data/cosmopedia.py', 'r', encoding='utf-8').read(),
    'data/__init__.py': '',
    'training/__init__.py': '',
}

# Create notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 NanoPrime - Standalone Kaggle Training\n",
                "\n",
                "**ZERO uploads needed!** All code embedded in this notebook.\n",
                "\n",
                "**Setup**: Enable GPU → Run All → Done!"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "!nvidia-smi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q transformers datasets einops"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": null,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create directory structure\n",
                "!mkdir -p src/modules data training\n",
                "print('✓ Directories created')"
            ]
        }
    ],
    "metadata": {
        "kaggle": {
            "accelerator": "gpu",
            "isGpuEnabled": True
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add cells for each file
for filepath, content in files_to_embed.items():
    cell = {
        "cell_type": "code",
        "execution_count": null,
        "metadata": {},
        "outputs": [],
        "source": [
            f"%%writefile {filepath}\n{content}"
        ]
    }
    notebook["cells"].append(cell)

# Save notebook
with open('kaggle_standalone.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("✓ Standalone notebook created!")
print("File: kaggle_standalone.ipynb")
print("Size:", len(json.dumps(notebook)) // 1024, "KB")
