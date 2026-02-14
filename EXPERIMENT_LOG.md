# DriveStudio 実験ログ

## 実験1: UV環境構築とgsplat 1.4.0移行（2026-02-14）

### 📋 実験概要

**目的**:
- drivestudioの環境をuvで再現可能にする
- gsplat 1.3.0 → 1.4.0へアップグレード
- PyTorch 2.0.0 → 2.1.2へアップグレード
- 参考リポジトリ（splatad, nuscenes-gs-lab）と環境を統一

**実施者**: masa
**日付**: 2026-02-14
**ステータス**: ✅ 環境構築完了、✅ データ前処理完了（8シーン）

---

## 🎯 実験設定

### 環境仕様

#### Before（元の環境）
```
Python: 3.9
PyTorch: 2.0.0+cu117
CUDA: 11.7
gsplat: 1.3.0 (GitHub直接インストール)
依存管理: pip + requirements.txt
```

#### After（構築した環境）
```
Python: 3.10
PyTorch: 2.1.2+cu118
CUDA: 11.8
gsplat: 1.4.0
依存管理: uv + pyproject.toml
```

### ハードウェア
- GPU: NVIDIA GeForce RTX 4070 Ti
- OS: Linux (Ubuntu)
- CUDA Version: 11.8

---

## 📝 実施内容

### Phase 1: 環境構築の計画立案

**参考リポジトリの調査**:
- `splatad`: torch 2.1.2 + CUDA 11.8を使用
- `nuscenes-gs-lab`: torch 2.1.2 + CUDA 11.8 + gsplat 1.4.0を使用

**決定事項**:
- Python 3.10（参考リポジトリと統一）
- PyTorch 2.1.2 + CUDA 11.8（動作実績あり）
- gsplat 1.4.0（最新版、ただしAPI変更あり）

**リスク分析**:
- ✅ PyTorch 2.0 → 2.1.2: 低リスク（互換性高い）
- ⚠️ gsplat 1.3.0 → 1.4.0: 中リスク（`cuda_legacy` API削除）

### Phase 2: pyproject.toml作成

**作成ファイル**: `pyproject.toml`

**重要な設定**:
```toml
[project]
requires-python = ">=3.10,<3.11"
dependencies = [
    "torch==2.1.2",
    "torchvision==0.16.2",
    "gsplat==1.4.0",
    # ... その他の依存関係
]

[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"

[[tool.uv.index]]
name = "gsplat-whl"
url = "https://docs.gsplat.studio/whl/pt21cu118"

[tool.uv]
override-dependencies = ["numpy<2"]
```

**工夫点**:
- カスタムインデックスでPyTorchとgsplatを指定
- numpy<2制約（互換性確保）
- pytorch3d、nvdiffrastは別途インストール（ビルド問題回避）

### Phase 3: gsplat 1.4.0 API対応

**問題**: gsplat 1.4.0で`cuda_legacy`モジュールが削除された

**影響範囲**: `models/gaussians/basics.py`

**修正内容**:

#### Before (gsplat 1.3.0)
```python
from gsplat.cuda_legacy._wrapper import num_sh_bases
from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
```

#### After (gsplat 1.4.0)
```python
# num_sh_bases の代替実装
def num_sh_bases(degree: int) -> int:
    """Calculate number of spherical harmonics bases for given degree"""
    return (degree + 1) ** 2

# quat_to_rotmat の代替
from gsplat.utils import normalized_quat_to_rotmat as quat_to_rotmat
```

**検証結果**:
- ✅ `num_sh_bases(3) = 16` (正しい計算結果)
- ✅ `quat_to_rotmat` 単位クォータニオン → 恒等行列変換成功

### Phase 4: 全依存関係のインストール

**インストール手順**:
```bash
# 1. 基本環境
uv venv --python 3.10
source .venv/bin/activate

# 2. メイン依存関係
uv sync

# 3. 追加パッケージ（手動）
uv pip install pip
uv pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
uv pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
uv pip install --no-build-isolation chumpy
```

**問題と解決**:
| 問題 | 原因 | 解決方法 |
|------|------|----------|
| pytorch3dビルド失敗 | torch依存がbuild-timeに必要 | `--no-build-isolation`使用 |
| chumpyビルド失敗 | pipモジュール必要 | 先にpipをインストール |
| setuptools設定エラー | 複数トップレベルパッケージ | `py-modules = []`設定 |

**インストール結果**:
- ✅ 全128パッケージ正常インストール
- ✅ PyTorch 2.1.2+cu118 動作確認
- ✅ gsplat 1.4.0 インポート成功
- ✅ drivestudioモジュール読み込み成功

---

## 🔬 検証実験

### 検証1: PyTorch環境確認

**スクリプト**: `test_phase1_pytorch.py`

**結果**:
```
✅ PyTorch version: 2.1.2+cu118
✅ torchvision version: 0.16.2+cu118
✅ CUDA available: True
✅ CUDA version: 11.8
✅ GPU count: 1
✅ Device name: NVIDIA GeForce RTX 4070 Ti
✅ GPU tensor operation successful
```

### 検証2: gsplat API互換性確認

**スクリプト**: `test_phase2_gsplat.py`, `test_phase2_imports.py`

**結果**:
```
✅ gsplat version: 1.4.0+pt21cu118
✅ gsplat.rendering.rasterization
✅ gsplat.cuda._wrapper.spherical_harmonics
✅ gsplat.utils.normalized_quat_to_rotmat
❌ gsplat.cuda_legacy (予想通り利用不可)
```

**代替実装の検証**:
```
✅ num_sh_bases(0) = 1
✅ num_sh_bases(1) = 4
✅ num_sh_bases(2) = 9
✅ num_sh_bases(3) = 16
✅ quat_to_rotmat: 単位クォータニオン → 恒等行列 (誤差 0.000000)
```

### 検証3: 全モジュールインポート確認

**スクリプト**: `test_phase4_imports.py`

**結果**:
```
✅ torch, torchvision, gsplat
✅ omegaconf, open3d, kornia, matplotlib, wandb
✅ pytorch3d, nvdiffrast
✅ models.gaussians.basics
✅ models.gaussians.vanilla
⚠️ chumpy (numpy互換性警告、動作は問題なし)
```

---

## 📊 データ前処理実験

### データセット: Waymo Open Dataset

**シーン数**: 8シーン
**ソース**: `data/waymo_example_scenes.txt`
**シーンID**: 23, 114, 327, 621, 703, 172, 552, 788

### 前処理設定

**コマンド**:
```bash
export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo \
    --split training \
    --split_file data/waymo_example_scenes.txt \
    --workers 8 \
    --process_keys images lidar calib pose dynamic_masks objects
```

**処理内容**:
- 画像抽出（5カメラ × 約200フレーム）
- LiDARデータ抽出
- カメラキャリブレーション
- カメラポーズ
- 動的マスク
- オブジェクトアノテーション

**パフォーマンス**:
- 並列処理: 8シーン同時
- 処理速度: 約1.7-2.9秒/フレーム
- 推定時間: 約10-15分

**進捗（完了）**: ✅
```
✅ 全8シーン処理完了
総処理時間: 約11分35秒（700秒）
終了コード: 0（正常終了）

処理速度:
- File 552: 8分45秒 (198フレーム) - 最速
- File 621: 10分19秒 (198フレーム)
- File 114: 10分20秒 (198フレーム)
- File 172: 10分22秒 (198フレーム)
- File 788: 10分37秒 (199フレーム)
- File 703: 11分4秒 (199フレーム)
- File 23: 11分20秒 (199フレーム)
- File 327: 11分35秒 (199フレーム) - 最遅
```

**出力確認**:
```bash
$ ls data/waymo/processed/training/
023  114  172  327  552  621  703  788

$ ls data/waymo/processed/training/023/
dynamic_masks/  ego_pose/  extrinsics/  frame_info.json  images/
instances/  intrinsics/  lidar/  sky_masks/

$ ls data/waymo/processed/training/023/images/ | wc -l
995  # 5カメラ × 199フレーム
```

**注意点**:
- libcudnn.so.8警告: データ前処理はCPUで実行、問題なし
- TensorFlow使用: waymo-open-dataset-tf-2-11-0==1.6.0

---

## 📁 作成ファイル

### 環境設定
- `pyproject.toml` - uv依存関係管理
- `.gitignore` - UV環境、参考リポジトリを除外

### ドキュメント
- `INSTALL_UV.md` - uvインストールガイド
- `UV_MIGRATION_PLAN.md` - 移行計画と完了記録
- `NEXT_STEPS.md` - 次のステップ計画
- `EXPERIMENT_LOG.md` - 本ファイル

### 検証スクリプト
- `test_phase1_pytorch.py` - PyTorch確認
- `test_phase2_gsplat.py` - gsplat API確認
- `test_phase2_imports.py` - gsplat互換性テスト
- `test_phase4_imports.py` - 全モジュール確認

### コード修正
- `models/gaussians/basics.py` - gsplat 1.4.0対応

---

## 🎯 次のステップ

### 短期（完了待ち）
- [x] Waymoデータ前処理完了確認 ✅
- [x] 前処理データの検証 ✅

### 中期（今後の実験）
1. **最初のトレーニング**
   ```bash
   python tools/train.py \
       --config_file configs/omnire.yaml \
       --output_root ./logs/test_omnire \
       --project first_test \
       --run_name scene_23_3cams \
       dataset=waymo/3cams \
       data.scene_idx=23 \
       data.start_timestep=0 \
       data.end_timestep=50
   ```
   - シーン: 23（最初のシーン）
   - カメラ: 3cams（軽量テスト）
   - フレーム: 0-50（約5秒分）

2. **Sky Masks抽出**（別環境必要）
   - SegFormer環境構築（PyTorch 1.8）
   - Sky masks抽出実行

3. **フルトレーニング**
   - 全フレーム、5カメラ
   - 8シーン全て

### 長期
- [ ] 論文結果の再現
- [ ] 他の手法（Deformable-GS、PVG）との比較
- [ ] 本家へのPR作成（gsplat 1.4.0対応）

---

## 💡 知見・メモ

### うまくいったこと
1. **段階的検証**: Phase 1-4で段階的に検証したことで、問題の早期発見が可能
2. **参考リポジトリの活用**: 動作実績のある環境設定をベースにしたことで、大きな問題を回避
3. **バックグラウンド実行**: データ前処理をバックグラウンドで実行し、効率化

### つまずいたこと
1. **gsplat API変更**: 事前調査で判明したが、公式ドキュメントでは明記されていなかった
2. **ビルド問題**: pytorch3d、nvdiffrast、chumpyで個別対応が必要
3. **pyproject.toml設定**: アプリケーション型プロジェクトの設定に試行錯誤

### 改善点
- [ ] chumpyをpyproject.tomlに含める方法を検討
- [ ] CI/CD用のスクリプト作成
- [ ] 環境構築の自動化

---

## 🔗 参考資料

### 本家リポジトリ
- [ziyc/drivestudio](https://github.com/ziyc/drivestudio)
- [OmniRe論文](https://arxiv.org/abs/2408.16760)

### 参考実装
- [splatad](https://github.com/user/splatad) - torch 2.1.2 + CUDA 11.8
- [nuscenes-gs-lab](https://github.com/user/nuscenes-gs-lab) - gsplat 1.4.0

### 技術資料
- [gsplat GitHub](https://github.com/nerfstudio-project/gsplat)
- [gsplat Documentation](https://docs.gsplat.studio/)
- [Waymo Open Dataset](https://waymo.com/open/)

---

---

## 🔧 トレーニング実験（2026-02-14 19:36-19:40）

### 実験2: OmniReトレーニング試行

**目的**: Waymo scene 23でOmniReの最初のトレーニングを実行

**設定**:
- シーン: 023
- カメラ: 3台（front, front_left, front_right）
- フレーム: 0-50（51フレーム）
- 手法: OmniRe（マルチ表現トレーナー）

**発生した問題**:

#### 問題1: Sky Masksが存在しない
**エラー**: `FileNotFoundError: sky_masks/000_0.png`

**原因**:
- Sky masksは前処理で自動生成されない
- SegFormer環境（PyTorch 1.8）で別途抽出が必要

**対処**: `data.pixel_source.load_sky_mask=false`を設定

**コード修正**: `datasets/base/pixel_source.py`
```python
# Before
if self.sky_masks is not None:

# After
if hasattr(self, 'sky_masks') and self.sky_masks is not None:
```

#### 問題2: SMPL人体ポーズデータが存在しない
**エラー**: `FileNotFoundError: humanpose/smpl.pkl`

**原因**:
- SMPL人体ポーズは別途処理が必要
- Google Driveからダウンロード可能（一部シーン）

**対処**: `data.pixel_source.load_smpl=false`を試行

#### 問題3: コードがSMPLデータ前提の実装
**エラー**: `AttributeError: 'WaymoPixelSource' object has no attribute 'smpl_human_all'`

**原因**:
- `driving_dataset.py:312`でSMPLデータの存在を前提
- SMPLなしでOmniReを実行するには、さらなるコード修正が必要

**結論**:
- ✅ 環境とデータの基本的な動作は確認できた
- ❌ OmniReは完全なデータセット（sky masks + SMPL）が必要
- 🔄 **Deformable-GSに切り替えて最初のトレーニングを実行**

---

## 📋 今後の実験計画

### 実験3: Deformable-GSでの最初のトレーニング

**ステータス**: 🚀 準備完了、実行待ち

**詳細計画**: [DEFORMABLE_GS_PLAN.md](DEFORMABLE_GS_PLAN.md)

**概要**:
- シンプルな手法で最初のトレーニング成功を確認
- Sky masks・SMPL不要で即座に開始可能
- 環境とデータの動作確認に最適

**実行コマンド**:
```bash
python tools/train.py \
    --config_file configs/deformablegs.yaml \
    --output_root ./logs/test_deformgs \
    --project first_test \
    --run_name scene_23_3cams \
    dataset=waymo/3cams \
    data.scene_idx=23 \
    data.start_timestep=0 \
    data.end_timestep=50 \
    data.pixel_source.load_sky_mask=false
```

---

### 実験4: OmniReでの高品質トレーニング（将来）

**ステータス**: 📝 データ準備が必要

**詳細計画**: [OMNIRE_PLAN.md](OMNIRE_PLAN.md)

**概要**:
- マルチ表現トレーナーで最高品質の再構築
- 完全なデータセット（Sky masks + SMPL）が必要
- 論文結果の再現を目指す

**必要な準備**:
1. SMPL人体ポーズのダウンロード（10分）
2. SegFormer環境構築（30分）
3. Sky masks抽出（30分）

**推定所要時間**: 約1.5時間

---

**更新履歴**:
- 2026-02-14 19:21: 実験開始、環境構築完了
- 2026-02-14 19:22: データ前処理開始（8シーン、バックグラウンド実行）
- 2026-02-14 19:33: データ前処理完了（総処理時間11分35秒）
- 2026-02-14 19:36-19:40: OmniReトレーニング試行、複数の問題に遭遇
- 2026-02-14 19:41: Deformable-GSに切り替え、実験計画を策定
