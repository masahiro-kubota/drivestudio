# OmniRe トレーニング計画

## 📋 概要

**目的**: OmniRe（マルチ表現トレーナー）でWaymo scene 023のトレーニングを成功させる

**作成日**: 2026-02-14
**ステータス**: 📝 データ準備が必要

---

## 🎯 OmniReとは

### 特徴

**マルチ表現トレーナー**:
- **背景**: 静的Gaussian（道路、建物など）
- **車両**: 静的Gaussian（RigidNodes）
- **人間**: SMPL-Gaussian（SMPLNodes）
- **その他動的物体**: 変形可能Gaussian（DeformableNodes）
- **空**: 環境光モデル（Sky）

### メリット

- ✅ **最高品質の再構築**（ICLR 2025 Spotlight）
- ✅ **論文結果の再現可能**
- ✅ **詳細なドキュメント**
- ✅ **公式実装で最も成熟**

### デメリット

- ❌ **複雑なデータ準備**（Sky masks、SMPL）
- ❌ **処理時間が長い**
- ❌ **複雑な構造**

---

## ⚠️ 必要なデータ

OmniReを実行するには、以下のデータが必要です：

### 1. ✅ 基本データ（準備済み）

- [x] 画像データ（5カメラ × 約200フレーム）
- [x] LiDARデータ
- [x] カメラキャリブレーション
- [x] カメラポーズ
- [x] 動的マスク
- [x] オブジェクトアノテーション

**場所**: `data/waymo/processed/training/023/`

### 2. ❌ Sky Masks（未準備）

**必要性**: 必須（空の再構築に使用）

**取得方法**: SegFormer環境で抽出

**状態**:
- ディレクトリは存在: `data/waymo/processed/training/023/sky_masks/`
- 中身は空（ファイルなし）

### 3. ❌ SMPL人体ポーズ（未準備）

**必要性**: 人体再構築を使用する場合に必須

**取得方法**:
- **Option A**: Google Driveからダウンロード（推奨）
- **Option B**: 自分で処理パイプラインを実行

**状態**:
- ディレクトリは存在しない: `data/waymo/processed/training/023/humanpose/`

---

## 📝 データ準備手順

### 準備1: SMPL人体ポーズのダウンロード（推奨）⭐

**所要時間**: 約5-10分

```bash
cd data

# Google Driveからダウンロード
gdown 1QrtMrPAQhfSABpfgQWJZA2o_DDamL_7_

# 解凍
unzip waymo_preprocess_humanpose.zip

# 不要なzipファイルを削除
rm waymo_preprocess_humanpose.zip

# 確認
ls waymo/processed/training/*/humanpose/
```

**期待される結果**:
```
waymo/processed/training/023/humanpose/smpl.pkl
waymo/processed/training/114/humanpose/smpl.pkl
...
```

### 準備2: Sky Masksの抽出

**所要時間**: 環境構築30分 + 抽出30分 = 約1時間

#### Step 1: SegFormer環境の構築

⚠️ **注意**: 別のconda環境が必要（PyTorch 1.8）

```bash
# 新しいconda環境を作成
conda create -n segformer python=3.8
conda activate segformer

# PyTorch 1.8をインストール
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 依存関係をインストール
pip install timm==0.3.2 pylint debugpy opencv-python-headless attrs ipython tqdm imageio scikit-image omegaconf

# mmcv-fullをインストール
pip install mmcv-full==1.2.7 --no-cache-dir

# SegFormerをクローン＆インストール
git clone https://github.com/NVlabs/SegFormer
cd SegFormer
pip install .
cd ..
```

#### Step 2: SegFormerモデルのダウンロード

```bash
# 作業ディレクトリに戻る
cd /home/masa/drivestudio

# モデルをダウンロード
gdown 1e7DECAH0TRtPZM6hTqRGoboq1XPqSmuj

# または手動でダウンロード:
# https://github.com/NVlabs/SegFormer#evaluation
# からsegformer.b5.1024x1024.city.160k.pthをダウンロード
```

#### Step 3: Sky Masksの抽出実行

```bash
# segformer環境を有効化
conda activate segformer

# マスク抽出を実行
python datasets/tools/extract_masks.py \
    --data_root data/waymo/processed/training \
    --segformer_path=./SegFormer \
    --checkpoint=./segformer.b5.1024x1024.city.160k.pth \
    --split_file data/waymo_example_scenes.txt \
    --process_dynamic_mask
```

**進捗確認**:
```bash
# 抽出されたマスクを確認
ls -lh data/waymo/processed/training/023/sky_masks/
```

**期待される結果**:
```
000_0.png  000_1.png  000_2.png  000_3.png  000_4.png
001_0.png  001_1.png  ...
```

---

## 🚀 トレーニング実行

### データ準備完了後の確認

```bash
# SMPL人体ポーズの確認
ls data/waymo/processed/training/023/humanpose/smpl.pkl

# Sky masksの確認
ls data/waymo/processed/training/023/sky_masks/*.png | wc -l
# 期待: 995ファイル（5カメラ × 199フレーム）

# すべてのデータが揃っているか確認
tree data/waymo/processed/training/023/ -L 1
```

### トレーニング実行コマンド

#### テストトレーニング（50フレーム）

```bash
export PYTHONPATH=$(pwd)
source .venv/bin/activate

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

**パラメータ説明**:
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `--config_file` | `configs/omnire.yaml` | OmniRe設定 |
| `--output_root` | `./logs/test_omnire` | ログ出力先 |
| `--project` | `first_test` | プロジェクト名 |
| `--run_name` | `scene_23_3cams` | 実行名 |
| `dataset` | `waymo/3cams` | 3カメラ構成 |
| `data.scene_idx` | `23` | シーン023 |
| `data.start_timestep` | `0` | 開始フレーム |
| `data.end_timestep` | `50` | 終了フレーム（51フレーム） |

#### フルトレーニング（全フレーム）

```bash
python tools/train.py \
    --config_file configs/omnire.yaml \
    --output_root ./logs/omnire_waymo \
    --project full_training \
    --run_name scene_23_5cams \
    dataset=waymo/5cams \
    data.scene_idx=23 \
    data.start_timestep=0 \
    data.end_timestep=-1
```

---

## 📊 進捗監視

### TensorBoard

```bash
# 別のターミナルで実行
tensorboard --logdir ./logs/test_omnire
# ブラウザで http://localhost:6006 にアクセス
```

### ログファイル

```bash
# リアルタイムでログを確認
tail -f ./logs/test_omnire/first_test/scene_23_3cams/logs.txt
```

### チェックポイント

```bash
# 定期的に確認
watch -n 60 'ls -lh ./logs/test_omnire/first_test/scene_23_3cams/checkpoints/'
```

---

## 📈 期待される結果

### トレーニング完了後

**出力ファイル**:
- ✅ チェックポイント: `checkpoints/step_*.ckpt`
- ✅ レンダリング結果: `renderings/`
- ✅ 設定ファイル: `config.yaml`
- ✅ ログ: `logs.txt`

**評価メトリクス**:
- PSNR > 25dB（高品質）
- SSIM > 0.8
- LPIPS < 0.2

**視覚的品質**:
- 🖼️ 背景の詳細な再構築
- 🚗 車両の高品質なレンダリング
- 🚶 人間の自然な動き（SMPL使用時）
- 🌌 空の美しいレンダリング（Sky masks使用時）

---

## 🚨 トラブルシューティング

### 問題1: Sky Masksが見つからない

**エラー**:
```
FileNotFoundError: sky_masks/000_0.png
```

**対処**:
1. Sky masksが抽出されているか確認
```bash
ls data/waymo/processed/training/023/sky_masks/
```

2. SegFormer環境でマスク抽出を実行
3. または一時的に無効化（非推奨）:
```bash
data.pixel_source.load_sky_mask=false
```

### 問題2: SMPL人体ポーズが見つからない

**エラー**:
```
FileNotFoundError: humanpose/smpl.pkl
```

**対処**:
1. SMPLデータがダウンロードされているか確認
```bash
ls data/waymo/processed/training/023/humanpose/smpl.pkl
```

2. Google Driveからダウンロード実行
3. または一時的に無効化（人体再構築なし）:
```bash
data.pixel_source.load_smpl=false
```
⚠️ ただし、コード修正が必要な場合あり

### 問題3: CUDA Out of Memory

**対処**:
1. カメラ数を減らす
```bash
dataset=waymo/3cams  # または 2cams
```

2. フレーム数を減らす
```bash
data.end_timestep=30
```

3. 解像度を下げる
```bash
data.pixel_source.downscale=2
```

### 問題4: SegFormer環境でエラー

**症状**: mmcv-fullのインストールエラー

**対処**:
```bash
# PyTorchバージョンを確認
python -c "import torch; print(torch.__version__)"
# 1.8.1であることを確認

# mmcv-fullを再インストール
pip uninstall mmcv-full
pip install mmcv-full==1.2.7 --no-cache-dir
```

---

## 📋 チェックリスト

### データ準備

- [ ] SMPL人体ポーズをダウンロード
  ```bash
  cd data && gdown 1QrtMrPAQhfSABpfgQWJZA2o_DDamL_7_ && unzip waymo_preprocess_humanpose.zip
  ```

- [ ] SegFormer環境を構築
  ```bash
  conda create -n segformer python=3.8 && conda activate segformer
  ```

- [ ] SegFormerモデルをダウンロード
  ```bash
  gdown 1e7DECAH0TRtPZM6hTqRGoboq1XPqSmuj
  ```

- [ ] Sky masksを抽出
  ```bash
  conda activate segformer && python datasets/tools/extract_masks.py ...
  ```

- [ ] データ完全性を確認
  ```bash
  ls data/waymo/processed/training/023/humanpose/smpl.pkl
  ls data/waymo/processed/training/023/sky_masks/*.png | wc -l
  ```

### トレーニング

- [ ] drivestudio環境を有効化
  ```bash
  source .venv/bin/activate
  ```

- [ ] テストトレーニング実行（50フレーム）
- [ ] 結果を確認
- [ ] フルトレーニング実行（全フレーム）
- [ ] 評価実行
- [ ] 結果をEXPERIMENT_LOG.mdに記録

---

## 🔄 代替案

### Option 1: SMPLのみダウンロード + Sky maskなし

**最速の方法**（約10分）:

```bash
# SMPLダウンロード
cd data
gdown 1QrtMrPAQhfSABpfgQWJZA2o_DDamL_7_
unzip waymo_preprocess_humanpose.zip
cd ..

# Sky maskなしでトレーニング
python tools/train.py \
    --config_file configs/omnire.yaml \
    --output_root ./logs/test_omnire \
    --project first_test \
    --run_name scene_23_3cams_no_sky \
    dataset=waymo/3cams \
    data.scene_idx=23 \
    data.start_timestep=0 \
    data.end_timestep=50 \
    data.pixel_source.load_sky_mask=false
```

**注意**: 空の再構築品質は低下

### Option 2: すべてのデータを準備（完璧）

**所要時間**: 約1.5時間

1. SMPLダウンロード（10分）
2. SegFormer環境構築（30分）
3. Sky masks抽出（30分）
4. トレーニング実行

**メリット**: 最高品質の結果

---

## 📚 参考資料

### 論文

- **OmniRe**: [Omni-Recon: Towards General-Purpose Neural Radiance Fields for Versatile 3D Applications](https://arxiv.org/abs/2408.16760)
- **SMPL**: [SMPL: A Skinned Multi-Person Linear Model](https://smpl.is.tue.mpg.de/)

### ドキュメント

- **Waymo準備ガイド**: [docs/Waymo.md](docs/Waymo.md)
- **Human Poseガイド**: [docs/HumanPose.md](docs/HumanPose.md)
- **本家リポジトリ**: [ziyc/drivestudio](https://github.com/ziyc/drivestudio)

### ツール

- **SegFormer**: [NVlabs/SegFormer](https://github.com/NVlabs/SegFormer)
- **gsplat**: [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)

---

## 🎯 次のステップ（成功後）

### 短期

1. 結果を評価・分析
2. Deformable-GSと比較
3. 他のシーンで実験

### 中期

1. 全8シーンでトレーニング
2. 論文結果の再現
3. 評価メトリクスの改善

### 長期

1. 手法の改善・拡張
2. 本家へのPR作成
3. 新しいデータセットで実験

---

**最終更新**: 2026-02-14 19:50
**次のアクション**: データ準備（SMPL + Sky masks）
**推定所要時間**: 1.5時間
