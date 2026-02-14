# DriveStudio - 次のステップ計画

## ✅ 完了した作業（2026-02-14）

### 環境構築
- ✅ Python 3.10 + uv環境
- ✅ PyTorch 2.1.2 + CUDA 11.8
- ✅ gsplat 1.4.0（cuda_legacy API代替実装済み）
- ✅ 全依存関係のインストール
- ✅ 動作確認完了

### データセット
- ✅ **Waymo Open Dataset**: 8シーン分ダウンロード済み
  - 場所: `data/waymo/raw/`
  - シーンリスト: `data/waymo_example_scenes.txt`
- ✅ **データ前処理完了**: 8シーン分処理済み
  - 出力先: `data/waymo/processed/training/`
  - 処理時間: 11分35秒

## 🎯 選択した環境

### データセット: Waymo Open Dataset
**理由**:
- ✅ 公式実装（OmniRe）で使用されているデータセット
- ✅ 既にダウンロード済み（8シーン）
- ✅ 論文の結果を再現可能
- ✅ 高品質な自動運転データ

### 推奨する手法の選択

#### オプション1: OmniRe（推奨）⭐
**特徴**:
- 公式実装（ICLR 2025 Spotlight）
- マルチ表現トレーナー
  - 背景：静的Gaussian
  - 車両：静的Gaussian
  - 人間：SMPL-Gaussian
  - その他：変形可能Gaussian
- **最も成熟した実装**

**メリット**:
- 論文の結果を再現可能
- 最高品質の再構築結果
- 詳細なドキュメントあり

**デメリット**:
- 複雑（マルチ表現）
- 処理時間がやや長い

#### オプション2: Deformable-GS（シンプル）
**特徴**:
- シングル表現トレーナー
- シーン全体を1つの変形可能Gaussianで表現
- シンプルな構造

**メリット**:
- 理解しやすい
- セットアップが簡単
- 高速

**デメリット**:
- OmniReより再構築品質が劣る可能性

### 🎯 推奨：まずOmniReで進める

理由：
1. 公式実装で最も成熟している
2. Waymoデータとの組み合わせで論文結果を再現可能
3. 環境構築が完了しているので、複雑さは問題ない
4. 最高品質の結果を得られる

## 📋 次のステップ（詳細計画）

### Step 1: Waymoデータの前処理 ⏳

#### 1-1. 前処理スクリプト実行
```bash
export PYTHONPATH=$(pwd)
source .venv/bin/activate

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
- 画像抽出
- LiDARデータ抽出
- カメラキャリブレーション
- カメラポーズ
- 動的マスク
- オブジェクトアノテーション

**予想時間**: 8シーン × 10-15分 = 約1.5-2時間

**確認**:
```bash
ls data/waymo/processed/training/
```

#### 1-2. Sky Masks抽出（必須）

⚠️ **注意**: SegFormerは別環境が必要（PyTorch 1.8）

**選択肢A**: SegFormer環境を作成して実行
```bash
# 別環境作成
conda create -n segformer python=3.8
conda activate segformer
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.2.7 --no-cache-dir

# SegFormerインストール
git clone https://github.com/NVlabs/SegFormer
cd SegFormer && pip install . && cd ..

# モデルダウンロード
gdown 1e7DECAH0TRtPZM6hTqRGoboq1XPqSmuj

# マスク抽出実行
python datasets/tools/extract_masks.py \
    --data_root data/waymo/processed/training \
    --segformer_path=./SegFormer \
    --checkpoint=./segformer.b5.1024x1024.city.160k.pth \
    --split_file data/waymo_example_scenes.txt \
    --process_dynamic_mask
```

**選択肢B**: Sky masksなしでトレーニング（非推奨）
- 空の再構築品質が低下
- 最初の試行としては許容可能

#### 1-3. Human Body Pose Processing（OmniRe用、オプション）

SMPLベースの人体再構築を使う場合のみ必要。

```bash
# 詳細はdocs/HumanPose.mdを参照
```

### Step 2: 最初のトレーニング実行 🚀

#### シナリオA: OmniRe（推奨）

**最小構成でテスト**:
```bash
export PYTHONPATH=$(pwd)
source .venv/bin/activate

# シーン23（最初のシーン）、3カメラで試す
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
- `scene_idx=23`: 最初のWaymoシーン
- `waymo/3cams`: 3カメラを使用（軽量）
- `end_timestep=50`: 最初の50フレームのみ（約5秒分）

**予想時間**:
- GPU: RTX 4070 Ti
- 推定: 30分〜1時間

#### シナリオB: Deformable-GS（シンプル）

```bash
python tools/train.py \
    --config_file configs/deformablegs.yaml \
    --output_root ./logs/test_deformgs \
    --project first_test \
    --run_name scene_23_3cams \
    dataset=waymo/3cams \
    data.scene_idx=23 \
    data.start_timestep=0 \
    data.end_timestep=50
```

### Step 3: 結果確認 📊

#### 3-1. TensorBoard確認
```bash
tensorboard --logdir ./logs/
```

#### 3-2. 評価実行
```bash
python tools/eval.py --resume_from ./logs/test_omnire/first_test/scene_23_3cams/checkpoints/latest.ckpt
```

#### 3-3. レンダリング結果確認
```bash
ls ./logs/test_omnire/first_test/scene_23_3cams/renderings/
```

### Step 4: フルトレーニング（成功後）

最初のテストが成功したら、フルパラメータでトレーニング：

```bash
# 全フレーム、5カメラ
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

## 🚨 想定される問題と対処

### 問題1: メモリ不足
**症状**: `CUDA out of memory`

**対処**:
- カメラ数を減らす（5cams → 3cams）
- バッチサイズを減らす
- 画像解像度を下げる

### 問題2: 前処理が遅い
**症状**: データ前処理に時間がかかる

**対処**:
- `--workers`を増やす（CPU コア数に応じて）
- 最初は1シーンのみで試す

### 問題3: SegFormer環境の問題
**症状**: PyTorch 1.8とPyTorch 2.1の環境の競合

**対処**:
- 完全に別のconda環境を作成
- Sky masksなしで最初は試す

## 📅 推奨スケジュール

### Day 1（今日）
- [x] 環境構築完了 ✅
- [x] Waymoデータ前処理実行（Step 1-1）✅ **完了: 8シーン、11分35秒**
- [ ] （オプション）Sky masks抽出（Step 1-2）

### Day 2
- [ ] 最初のトレーニング実行（Step 2: scene 23、50フレーム）
- [ ] 結果確認（Step 3）

### Day 3以降
- [ ] フルトレーニング（Step 4）
- [ ] 他のシーンで実験
- [ ] 異なる手法の比較

## 🎯 最終目標

1. **短期目標**: Waymo scene 23でOmniReのトレーニング成功
2. **中期目標**: 8シーン全てで論文結果を再現
3. **長期目標**: 手法の改善・新機能の追加

## 📝 次のアクション

**✅ 完了した作業**:
- ✅ 環境構築完了
- ✅ データ前処理完了（8シーン）
- ✅ OmniRe試行と問題分析

**🚀 次のステップ: Deformable-GSでトレーニング実行**

⚠️ **重要**: OmniReからDeformable-GSに切り替えました（詳細: [DEFORMABLE_GS_PLAN.md](DEFORMABLE_GS_PLAN.md)）

**理由**:
- OmniReはSky masks + SMPL人体ポーズが必要
- Deformable-GSはシンプルで最初のテストに最適

```bash
export PYTHONPATH=$(pwd)
source .venv/bin/activate

# Deformable-GSでトレーニング
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

**前処理データ確認**:
```bash
# 前処理されたデータを確認
ls -lh data/waymo/processed/training/
ls -lh data/waymo/processed/training/023/
```

**進捗監視**:
```bash
# TensorBoard起動
tensorboard --logdir ./logs/test_deformgs
```

---

**作成日**: 2026-02-14
**最終更新**: 2026-02-14 19:45
**ステータス**: ✅ 環境構築完了、✅ データ前処理完了（8シーン）、⚠️ OmniRe試行→問題発見 → 🚀 Deformable-GSでトレーニング準備完了

## 📚 関連ドキュメント

- **詳細計画**: [DEFORMABLE_GS_PLAN.md](DEFORMABLE_GS_PLAN.md) - Deformable-GSの完全な実行計画
- **実験ログ**: [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) - すべての実験記録と問題の詳細
- **インストール**: [INSTALL_UV.md](INSTALL_UV.md) - 環境構築手順
