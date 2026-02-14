# DriveStudio UV化 移行計画

## 📋 目的
drivestudioの環境構築をuvで再現可能にする。動作実績のある他リポジトリ（splatad, nuscenes-gs-lab）の設定を参考にしながら、段階的に検証していく。

## 🔍 現状分析

### 現在のdrivestudio環境（requirements.txt）
- **PyTorch**: `torch==2.0.0+cu117`
- **torchvision**: `torchvision==0.15.0+cu117`
- **gsplat**: インストール方法: `pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0`
- **xformers**: `xformers==0.0.18`
- **Python**: 指定なし（README.mdでは3.9）
- **CUDA**: 11.7

### 参考リポジトリの環境

#### splatad
```toml
Python: >=3.10,<3.11
torch: 2.1.2 (CUDA 11.8)
torchvision: 0.16.2
gsplat: カスタム実装（内部フォーク）
```

#### nuscenes-gs-lab
```toml
Python: >=3.10,<3.11
torch: 2.1.2 (CUDA 11.8)
torchvision: 0.16.2
gsplat: 1.4.0
```

### 🎯 目標環境（合わせる）
- **Python**: 3.10
- **PyTorch**: 2.1.2 + CUDA 11.8
- **torchvision**: 0.16.2
- **gsplat**: 1.4.0（要検証）

## ⚠️ 懸念事項

### gsplat 1.3.0 → 1.4.0 移行の不確実性

**drivestudioのコード（models/gaussians/basics.py:12-14）**
```python
from gsplat.cuda_legacy._wrapper import num_sh_bases
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
```

**問題**: `gsplat.cuda_legacy` モジュールがgsplat 1.4.0で利用可能かどうか不明

**対応**:
- まず gsplat 1.4.0 をインストールして実際に確認する
- 利用可能なら → そのまま使用
- 利用不可なら → 代替実装を検討

## 📝 段階的実装計画

### Phase 0: 準備
- [x] 参考リポジトリのpyproject.tomlを確認
- [x] drivestudioの依存関係を分析
- [ ] 計画をマークダウンに整理（このファイル）

### Phase 1: pyproject.toml作成
**目標**: 基本的なpyproject.tomlを作成

#### 1-1. 最小構成でpyproject.tomlを作成
```toml
[project]
name = "drivestudio"
version = "0.1.0"
description = "DriveStudio: 3DGS framework for autonomous driving"
requires-python = ">=3.10,<3.11"
dependencies = [
    "torch==2.1.2",
    "torchvision==0.16.2",
]

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu118" }
torchvision = { index = "pytorch-cu118" }

[tool.uv]
override-dependencies = [
    "numpy<2",
]
```

#### 1-2. PyTorchのみの環境で検証
```bash
# 最小環境構築
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .

# 検証コード実行
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

**期待結果**: PyTorch 2.1.2 + CUDA 11.8が正しくインストールされる

### Phase 2: gsplat検証
**目標**: gsplat 1.4.0の互換性を確認

#### 2-1. gsplat 1.4.0を追加
```toml
dependencies = [
    "torch==2.1.2",
    "torchvision==0.16.2",
    "gsplat==1.4.0",
]

[[tool.uv.index]]
name = "gsplat-whl"
url = "https://docs.gsplat.studio/whl/pt21cu118"
explicit = true

[tool.uv.sources]
gsplat = { index = "gsplat-whl" }
```

#### 2-2. cuda_legacy API確認スクリプト
```python
# test_gsplat_api.py
import gsplat

print(f"gsplat version: {gsplat.__version__}")

# cuda_legacy APIの確認
try:
    from gsplat.cuda_legacy._wrapper import num_sh_bases
    print("✅ num_sh_bases: 利用可能")
except ImportError as e:
    print(f"❌ num_sh_bases: {e}")

try:
    from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
    print("✅ quat_to_rotmat: 利用可能")
except ImportError as e:
    print(f"❌ quat_to_rotmat: {e}")

# 代替APIの確認
try:
    from gsplat.rendering import rasterization
    print("✅ rasterization: 利用可能")
except ImportError as e:
    print(f"❌ rasterization: {e}")

try:
    from gsplat.cuda._wrapper import spherical_harmonics
    print("✅ spherical_harmonics: 利用可能")
except ImportError as e:
    print(f"❌ spherical_harmonics: {e}")
```

**実行**:
```bash
uv pip install -e .
python test_gsplat_api.py
```

**分岐**:
- **ケースA**: cuda_legacy API が使える → Phase 3へ進む
- **ケースB**: cuda_legacy API が使えない → Phase 2-3で代替実装

#### 2-3. (ケースBのみ) 代替実装の検討
```python
# 代替案1: num_sh_bases
def num_sh_bases(degree: int) -> int:
    """Calculate number of spherical harmonics bases"""
    return (degree + 1) ** 2

# 代替案2: quat_to_rotmat
# Option A: gsplatの別モジュールを使用
from gsplat.utils import normalized_quat_to_rotmat

# Option B: pytorch3dを使用
from pytorch3d.transforms import quaternion_to_matrix
```

### Phase 3: 全依存関係の追加
**目標**: requirements.txtの全パッケージをpyproject.tomlに移植

#### 3-1. 依存関係を分類
```toml
dependencies = [
    # Core ML
    "torch==2.1.2",
    "torchvision==0.16.2",
    "gsplat==1.4.0",
    "timm==0.9.5",
    "pytorch_msssim==1.0.0",

    # Configuration
    "omegaconf==2.3.0",
    "torchmetrics==0.10.3",

    # Logging/Visualization
    "tensorboard==2.11.0",
    "wandb==0.15.8",
    "matplotlib>=3.8",  # override-dependenciesで指定
    "plotly==5.13.1",
    "viser==0.2.1",

    # Image/Video
    "imageio",
    "imageio-ffmpeg",
    "scikit-image==0.20.0",
    "opencv-python",

    # 3D Processing
    "open3d==0.16.0",
    "pyquaternion==0.9.9",
    "chumpy",
    "numpy<2",  # override-dependenciesで制約
    "kornia==0.7.2",

    # Utilities
    "tqdm",
    "gdown",
    "nerfview==0.0.3",
    "lpips==0.1.4",
]
```

#### 3-2. xformersの扱い
- **調査結果**: コード内でxformersの直接的なインポートが見つからない
- **決定**: いったん除外し、必要になったら追加

#### 3-3. GitHub経由パッケージの対応
**requirements.txtの指定**:
```txt
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast
```

**pyproject.tomlでの対応**:
```toml
dependencies = [
    # ... 他の依存関係 ...
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git",
    "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git",
]
```

### Phase 4: 動作確認
**目標**: 実際のコードで動作確認

#### 4-1. 簡単なインポートテスト
```python
# test_imports.py
"""drivestudioの主要モジュールがインポートできるか確認"""

import torch
import torchvision
import gsplat
import omegaconf
import open3d
import kornia

print("✅ All core imports successful")

# gsplat APIテスト
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import spherical_harmonics

print("✅ gsplat APIs available")

# drivestudioモジュール
from models.gaussians.basics import random_quat_tensor
from models.losses import l1_loss

print("✅ drivestudio modules loadable")
```

#### 4-2. 最小データでの動作テスト
```bash
# 設定ファイルのバリデーション
python -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('configs/deformablegs.yaml'); print('✅ Config loaded')"

# モデルの初期化テスト（データ不要）
# TODO: 具体的なテストコードを作成
```

### Phase 5: ドキュメント整備
- [ ] README.mdにuv用のインストール手順を追加
- [ ] pyproject.tomlにコメントを追加
- [ ] 移行完了後、このPLAN.mdを更新

## 🚨 注意事項

### リスク管理
1. **元のrequirements.txtは残す**: 後方互換性のため
2. **段階的に進める**: 各Phaseで検証を挟む
3. **バックアップ**: 環境構築前にプロジェクトをバックアップ

### バージョンの柔軟性
- 固定すべき: torch, torchvision, gsplat（互換性重要）
- 柔軟にできる: matplotlib, numpy（override-dependenciesで制約のみ）

## 📊 進捗管理

### ✅ 完了したタスク（2026-02-14）
- [x] 参考リポジトリ（splatad, nuscenes-gs-lab）の調査
- [x] drivestudioの依存関係分析
- [x] gsplat APIの使用状況調査
- [x] Phase 1-1: 最小構成のpyproject.toml作成
- [x] Phase 1-2: PyTorchのみの環境で検証
- [x] Phase 2-1: gsplat 1.4.0を追加
- [x] Phase 2-2: cuda_legacy API確認スクリプト実行
- [x] Phase 2-3: cuda_legacy API代替実装（models/gaussians/basics.py修正）
- [x] Phase 3: 全依存関係の追加
- [x] Phase 4: 実際のコードで動作確認

### 🎉 完了状況

**UV環境構築が完全に成功しました！**

#### 達成された環境
- Python 3.10
- PyTorch 2.1.2 + CUDA 11.8
- gsplat 1.4.0（cuda_legacy API代替実装済み）
- すべての依存関係（pytorch3d, nvdiffrast, chumpy含む）

#### 実施した主な変更
1. **pyproject.toml作成**: uv対応の依存関係管理
2. **models/gaussians/basics.py修正**: gsplat 1.4.0対応
   - `num_sh_bases`: 独自実装
   - `quat_to_rotmat`: `gsplat.utils.normalized_quat_to_rotmat`使用
3. **検証スクリプト作成**: Phase 1-4の各段階で動作確認

## 🔗 参考資料
- [splatad/pyproject.toml](/home/masa/splatad/pyproject.toml)
- [nuscenes-gs-lab/pyproject.toml](/home/masa/nuscenes-gs-lab/pyproject.toml)
- [drivestudio/requirements.txt](requirements.txt)
- [gsplat GitHub](https://github.com/nerfstudio-project/gsplat)
- [gsplat Documentation](https://docs.gsplat.studio/)
