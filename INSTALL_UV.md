# DriveStudio - UV インストールガイド

## 🚀 クイックスタート

```bash
# 1. 仮想環境作成
uv venv --python 3.10

# 2. 環境を有効化
source .venv/bin/activate

# 3. 依存関係をインストール
uv sync

# 4. 追加パッケージをインストール（ビルドの問題を回避）
uv pip install pip
uv pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
uv pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
uv pip install --no-build-isolation chumpy

# 5. SMPL Gaussians用のセットアップ
cd third_party/smplx/
pip install -e .
cd ../..
```

## 📦 インストールされるパッケージ

### コア
- **Python**: 3.10
- **PyTorch**: 2.1.2 + CUDA 11.8
- **torchvision**: 0.16.2
- **gsplat**: 1.4.0 ⚠️ **cuda_legacy API代替実装済み**

### 主要依存関係
- omegaconf, tensorboard, wandb（設定・ログ）
- open3d, pytorch3d, kornia（3D処理）
- matplotlib, plotly, viser（可視化）
- その他、requirements.txtの全パッケージ

## ⚙️ 環境詳細

### PyTorch + CUDA
```bash
torch==2.1.2+cu118
torchvision==0.16.2+cu118
CUDA 11.8
```

### gsplat 1.4.0 対応

**重要**: このプロジェクトはgsplat 1.4.0に対応済みです。

**変更点**:
- gsplat 1.3.0の`cuda_legacy`モジュールは1.4.0で削除されました
- [models/gaussians/basics.py](models/gaussians/basics.py)で代替実装を使用：
  - `num_sh_bases`: 独自実装 `(degree + 1) ** 2`
  - `quat_to_rotmat`: `gsplat.utils.normalized_quat_to_rotmat`を使用

## 🔍 動作確認

```bash
# Phase 1: PyTorch確認
python test_phase1_pytorch.py

# Phase 2: gsplat確認
python test_phase2_gsplat.py

# Phase 4: 全モジュール確認
python test_phase4_imports.py
```

## ⚠️ トラブルシューティング

### chumpy の警告
```
FutureWarning: In the future `np.bool` will be defined as the corresponding NumPy scalar.
```
→ 古いパッケージのため警告が出ますが、動作に問題ありません。

### pytorch3d / nvdiffrast のビルドエラー
→ `--no-build-isolation`オプション付きでインストールしてください（上記手順参照）

## 📚 参考

- **元のインストール方法**: `pip install -r requirements.txt`
- **gsplat 1.3.0との違い**: [UV_MIGRATION_PLAN.md](UV_MIGRATION_PLAN.md)参照
- **参考プロジェクト**:
  - [splatad](https://github.com/user/splatad)
  - [nuscenes-gs-lab](https://github.com/user/nuscenes-gs-lab)

## ✅ 完了チェックリスト

- [ ] uv環境作成完了
- [ ] すべての依存関係インストール完了
- [ ] PyTorch + CUDA動作確認
- [ ] gsplat 1.4.0動作確認
- [ ] drivestudioモジュールインポート確認
- [ ] (オプション) データセットの準備

---

**作成日**: 2026-02-14
**gsplat対応**: v1.3.0 → v1.4.0
**PyTorch**: 2.0.0+cu117 → 2.1.2+cu118
