# Deformable-GS トレーニング計画

## 📋 概要

**目的**: Deformable-GSでWaymo scene 023の最初のトレーニングを成功させる

**作成日**: 2026-02-14
**ステータス**: 🚀 準備完了、実行待ち

---

## 🎯 なぜDeformable-GSか

### OmniReで発生した問題

1. ❌ **Sky Masks不足**
   - SegFormer環境（PyTorch 1.8）での別途処理が必要
   - 時間がかかる

2. ❌ **SMPL人体ポーズ不足**
   - Google Driveからダウンロード可能だが、追加ステップが必要
   - コードがSMPLデータ前提で実装されている

3. ❌ **コード修正の複雑さ**
   - SMPLなしで動作させるには大幅な修正が必要

### Deformable-GSの利点

1. ✅ **シンプルな前提条件**
   - Sky masks不要
   - SMPL人体ポーズ不要
   - 基本的なデータ（画像、LiDAR、マスク）のみで動作

2. ✅ **理解しやすい構造**
   - シングル表現トレーナー
   - シーン全体を1つの変形可能Gaussianで表現

3. ✅ **最初のテストに最適**
   - 環境の動作確認
   - データの検証
   - パイプライン全体の理解

---

## 🚀 実行手順

### ステップ1: 環境確認

```bash
# 仮想環境を有効化
source .venv/bin/activate

# 環境変数設定
export PYTHONPATH=$(pwd)

# GPU確認
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### ステップ2: 設定ファイル確認

```bash
# Deformable-GS設定
cat configs/deformablegs.yaml | head -50

# Waymo 3カメラ設定
cat configs/datasets/waymo/3cams.yaml
```

### ステップ3: トレーニング実行

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

**パラメータ説明**:
| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `--config_file` | `configs/deformablegs.yaml` | Deformable-GS設定 |
| `--output_root` | `./logs/test_deformgs` | ログ出力先 |
| `--project` | `first_test` | プロジェクト名 |
| `--run_name` | `scene_23_3cams` | 実行名 |
| `dataset` | `waymo/3cams` | 3カメラ構成 |
| `data.scene_idx` | `23` | シーン023 |
| `data.start_timestep` | `0` | 開始フレーム |
| `data.end_timestep` | `50` | 終了フレーム（51フレーム） |
| `data.pixel_source.load_sky_mask` | `false` | Sky masksを使用しない |

### ステップ4: 進捗監視

**オプション1: TensorBoard**
```bash
# 別のターミナルで実行
tensorboard --logdir ./logs/test_deformgs
# ブラウザで http://localhost:6006 にアクセス
```

**オプション2: ログファイル**
```bash
# リアルタイムでログを確認
tail -f ./logs/test_deformgs/first_test/scene_23_3cams/logs.txt
```

**オプション3: 定期的な確認**
```bash
# 進捗確認スクリプト
watch -n 60 'ls -lh ./logs/test_deformgs/first_test/scene_23_3cams/checkpoints/'
```

### ステップ5: 結果確認

```bash
# チェックポイント確認
ls -lh ./logs/test_deformgs/first_test/scene_23_3cams/checkpoints/

# レンダリング結果
ls -lh ./logs/test_deformgs/first_test/scene_23_3cams/renderings/

# 設定ファイルのバックアップ
cat ./logs/test_deformgs/first_test/scene_23_3cams/config.yaml
```

### ステップ6: 評価実行

```bash
# 最新チェックポイントで評価
python tools/eval.py \
    --resume_from ./logs/test_deformgs/first_test/scene_23_3cams/checkpoints/latest.ckpt

# または特定のイテレーション
python tools/eval.py \
    --resume_from ./logs/test_deformgs/first_test/scene_23_3cams/checkpoints/step_30000.ckpt
```

---

## 📊 期待される結果

### トレーニング進行

**初期段階（0-1,000イテレーション）**:
- Gaussian初期化（LiDARポイントから）
- Loss値が急速に減少
- 大まかな形状が見え始める

**中間段階（1,000-15,000イテレーション）**:
- Gaussianの分割・統合
- 詳細が徐々に改善
- Loss値が安定して減少

**最終段階（15,000-30,000イテレーション）**:
- 細部の調整
- Loss値が収束
- 高品質なレンダリング

### 成功の指標

**必須**:
- ✅ トレーニングが完了（30,000イテレーション）
- ✅ エラーなく実行
- ✅ チェックポイントが保存される

**品質**:
- 📈 Loss値が順調に減少（RGB Loss < 0.1）
- 📈 PSNR > 20dB（目安）
- 📈 SSIM > 0.7（目安）

**視覚的**:
- 🖼️ レンダリング画像が生成される
- 🖼️ 車両や建物の形状が認識できる
- 🖼️ 色とテクスチャが妥当

---

## 🚨 トラブルシューティング

### 問題1: CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory
```

**対処法（優先度順）**:

1. **フレーム数を減らす**
```bash
data.end_timestep=30  # 50 → 30フレーム
```

2. **カメラ数を減らす**
```bash
dataset=waymo/2cams  # 3カメラ → 2カメラ
```

3. **解像度を下げる**
```bash
data.pixel_source.downscale=2  # より低解像度
```

4. **バッチサイズを調整**（設定ファイル編集が必要）

### 問題2: データ読み込みエラー

**症状**:
```
FileNotFoundError: ...
AttributeError: ...
```

**対処法**:
1. データの存在確認
```bash
ls data/waymo/processed/training/023/
```

2. PYTHONPATHの確認
```bash
echo $PYTHONPATH
```

3. エラーメッセージを詳細に確認

### 問題3: Loss値が下がらない

**症状**:
- 数千イテレーション経過してもLoss > 0.5
- レンダリング結果が改善しない

**対処法**:
1. **学習率を確認**（ログから）
2. **Gaussian数を確認**（少なすぎる/多すぎる）
3. **初期化を確認**（LiDARポイント数）
4. **設定を見直す**（別の設定ファイルを試す）

### 問題4: トレーニングが途中で停止

**症状**:
- プロセスが予期せず終了
- ログが更新されない

**対処法**:
1. **GPUの状態確認**
```bash
nvidia-smi
```

2. **ディスク容量確認**
```bash
df -h
```

3. **最後のログを確認**
```bash
tail -100 ./logs/test_deformgs/first_test/scene_23_3cams/logs.txt
```

4. **チェックポイントから再開**
```bash
python tools/train.py \
    --resume_from ./logs/test_deformgs/first_test/scene_23_3cams/checkpoints/latest.ckpt
```

---

## 📈 次のステップ（成功後）

### 短期（今日〜明日）

1. **結果を分析**
   - レンダリング品質を評価
   - メトリクスを確認（PSNR, SSIM, LPIPS）
   - 問題点を特定

2. **設定を最適化**
   - 必要に応じてパラメータ調整
   - 再トレーニング

### 中期（今週）

1. **フルフレームでトレーニング**
```bash
data.start_timestep=0
data.end_timestep=-1  # 全フレーム（199フレーム）
```

2. **5カメラでトレーニング**
```bash
dataset=waymo/5cams
```

3. **他のシーンで実験**
```bash
data.scene_idx=114  # シーン114
data.scene_idx=327  # シーン327
```

### 長期（今後）

1. **OmniReのデータを準備**
   - Option A: SMPL人体ポーズをダウンロード
   ```bash
   cd data
   gdown 1QrtMrPAQhfSABpfgQWJZA2o_DDamL_7_
   unzip waymo_preprocess_humanpose.zip
   ```

   - Option B: Sky masksを抽出
     - SegFormer環境構築
     - マスク抽出実行

2. **OmniReでトレーニング**
   - マルチ表現の利点を活用
   - より高品質な結果を目指す

3. **手法の比較**
   - Deformable-GS vs OmniRe
   - 定量的・定性的評価
   - 論文結果との比較

4. **本家へのコントリビュート**
   - gsplat 1.4.0対応のPR作成
   - ドキュメント改善
   - Issue報告

---

## 📝 記録すべき情報

### トレーニング中

- [ ] 開始時刻
- [ ] GPU使用率（`nvidia-smi`）
- [ ] メモリ使用量
- [ ] 推定完了時刻

### トレーニング後

- [ ] 完了時刻
- [ ] 総所要時間
- [ ] 最終Loss値
- [ ] PSNR, SSIM, LPIPS
- [ ] チェックポイントサイズ
- [ ] 問題点と改善点

### スクリーンショット

- [ ] TensorBoardのLossグラフ
- [ ] レンダリング結果（数枚）
- [ ] Ground truthとの比較

---

## 🎓 学習リソース

### Deformable-GSについて

- **論文**: [Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction](https://arxiv.org/abs/2309.13101)
- **コンセプト**: 各Gaussianに変形パラメータを追加し、動的シーンを表現

### 3D Gaussian Splattingの基礎

- **原論文**: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079)
- **gsplat**: [GitHub - nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat)

### Waymo Open Dataset

- **公式サイト**: [Waymo Open Dataset](https://waymo.com/open/)
- **ドキュメント**: [Waymo Dataset Format](https://github.com/waymo-research/waymo-open-dataset)

---

---

## ✅ 実行完了（2026-02-14 21:54）

### 🎉 結果サマリー

**Scene 023、3カメラ、0-50フレーム（51フレーム）**

| 指標 | 初期値 | 最終値 | 改善 |
|------|--------|--------|------|
| PSNR | 12.34 dB | **27.58 dB** | +15.24 dB |
| SSIM | 0.5850 | **0.9173** | +0.3323 |
| LPIPS | 0.8084 | **0.1114** | -0.6970 |

**成功要因**:
- ✅ ダミーsky masks生成で問題回避
- ✅ マルチカメラ（3台）でステレオ視差を獲得
- ✅ LiDAR深度情報（312万点）
- ✅ 30,000イテレーション完了（約27分）

### ⚠️ 重要な発見：自車はほぼ停止中

**Scene 023の移動状況**:
```
総移動距離: 0.01 m (1cm)
所要時間: 5.1秒
平均速度: 0.0 km/h
→ ほぼ停止中（視差なし）
```

**それでも良い結果が得られた理由**:
1. マルチカメラによるステレオ視差
2. LiDAR深度情報
3. 動的オブジェクト（他の車両・歩行者）

### 📈 次の実験計画

より良い結果を得るため、**走行中のシーン**でトレーニングを実施する。

---

## 🚗 次回実験：走行中シーンでのトレーニング

### 目的

**視差の影響を検証**し、走行中シーンでの性能向上を確認する。

### シーン選定

まず、各シーンの移動距離を確認：

```bash
python scripts/analyze_scene_motion.py
```

**候補シーン**（`data/waymo_example_scenes.txt`より）:
- Scene 114 (`seg125050`)
- Scene 327 (`seg169514`)
- Scene 172 (`seg138251`, frames 30-180)

### 実験1：Scene 114（走行中シーン）

**設定**:
```bash
export PYTHONPATH=$(pwd)
source .venv/bin/activate

python tools/train.py \
    --config_file configs/deformablegs.yaml \
    --output_root ./logs/deformgs_moving \
    --project scene_comparison \
    --run_name scene_114_3cams \
    dataset=waymo/3cams \
    data.scene_idx=114 \
    data.start_timestep=0 \
    data.end_timestep=50 \
    data.pixel_source.load_smpl=false
```

**期待される結果**:
- 視差による深度推定の改善
- PSNR > 28 dB（scene 023より向上）
- 動的オブジェクトの再現性向上

### 実験2：5カメラ構成での比較

**目的**: カメラ数の影響を検証

```bash
# Scene 114、5カメラ
python tools/train.py \
    --config_file configs/deformablegs.yaml \
    --output_root ./logs/deformgs_moving \
    --project scene_comparison \
    --run_name scene_114_5cams \
    dataset=waymo/5cams \
    data.scene_idx=114 \
    data.start_timestep=0 \
    data.end_timestep=50 \
    data.pixel_source.load_smpl=false
```

**比較項目**:
| 条件 | Scene | カメラ | 期待PSNR |
|------|-------|--------|----------|
| ベースライン | 023（停止） | 3 | 27.58 dB |
| 実験1 | 114（走行） | 3 | > 28 dB |
| 実験2 | 114（走行） | 5 | > 29 dB |

### 実験3：フルシーケンス（199フレーム）

**走行距離が最大化**される長いシーケンスで学習：

```bash
python tools/train.py \
    --config_file configs/omnire_extended_cam.yaml \
    --output_root ./logs/deformgs_full \
    --project full_sequence \
    --run_name scene_114_5cams_full \
    dataset=waymo/5cams \
    data.scene_idx=114 \
    data.start_timestep=0 \
    data.end_timestep=-1 \
    data.pixel_source.load_smpl=false
```

**注意**: 画像数が多い（199×5=995枚）ため、`omnire_extended_cam.yaml`を使用。

### データ準備スクリプト

各シーンの移動距離を確認するスクリプトを作成：

```bash
# scripts/analyze_scene_motion.py
python3 << 'EOF'
import numpy as np
import os

scenes = [23, 114, 172, 327, 552, 621, 703, 788]
results = []

for scene_idx in scenes:
    scene_dir = f'data/waymo/processed/training/{scene_idx:03d}'
    if not os.path.exists(f'{scene_dir}/ego_pose'):
        continue

    # 最初の50フレームの移動距離を計算
    poses = []
    for i in range(min(51, len(os.listdir(f'{scene_dir}/ego_pose')))):
        pose = np.loadtxt(f'{scene_dir}/ego_pose/{i:03d}.txt')
        poses.append(pose[:3, 3])

    if len(poses) < 2:
        continue

    poses = np.array(poses)
    distances = np.linalg.norm(np.diff(poses, axis=0), axis=1)
    total_dist = np.sum(distances)
    avg_speed = total_dist / (len(poses) / 10) if len(poses) > 1 else 0

    results.append({
        'scene': scene_idx,
        'frames': len(poses),
        'distance': total_dist,
        'speed_ms': avg_speed,
        'speed_kmh': avg_speed * 3.6
    })

# ソートして表示
results.sort(key=lambda x: x['distance'], reverse=True)

print("シーン別移動距離（0-50フレーム）:")
print("-" * 70)
print(f"{'Scene':<8} {'Frames':<8} {'Distance(m)':<15} {'Speed(km/h)':<12} {'推奨'}")
print("-" * 70)

for r in results:
    recommend = "✅ 推奨" if r['distance'] > 10 else ("⚠️  低速" if r['distance'] > 1 else "❌ 停止")
    print(f"{r['scene']:<8} {r['frames']:<8} {r['distance']:<15.2f} {r['speed_kmh']:<12.1f} {recommend}")
EOF
```

### 評価と比較

すべての実験完了後：

```bash
# 結果を収集
python utils/gather_results.py \
    --log_dirs logs/test_deformgs/first_test/scene_23_3cams \
                logs/deformgs_moving/scene_comparison/scene_114_3cams \
                logs/deformgs_moving/scene_comparison/scene_114_5cams
```

**分析ポイント**:
1. 視差の有無による性能差
2. カメラ数の影響
3. シーケンス長の影響
4. 動的オブジェクトの再現品質

---

## 📦 追加機能：PLYエクスポート（時系列対応）

### PLYと動的シーンの関係

**重要な理解**:

PLYファイルは**静的**ですが、Deformable-GSでは**時刻ごとに異なるPLY**を生成できます。

```
基本Gaussians (canonical space)
       ↓
   [時刻 t=0]  → Deformation Network → PLY (t=0)
   [時刻 t=25] → Deformation Network → PLY (t=25)
   [時刻 t=50] → Deformation Network → PLY (t=50)
```

**動的な動きの仕組み**:

1. **Canonical Gaussians**（基準となるGaussian配置）
   - チェックポイントに保存されている基本位置

2. **Deformation Network**（変形ネットワーク）
   - 入力: (Gaussian位置, 時刻t)
   - 出力: 変形後の位置・回転・スケール

3. **時刻tでのPLY**
   ```python
   # 時刻tでの変形を適用
   deformed_positions = canonical_positions + deformation(t)
   deformed_rotations = canonical_rotations * deformation_rot(t)
   ```

### 実装タスク

#### タスク1: 時系列PLYエクスポートスクリプト

```bash
# scripts/export_ply_sequence.py
python scripts/export_ply_sequence.py \
    --checkpoint logs/test_deformgs/first_test/scene_23_3cams/checkpoint_final.pth \
    --output_dir scene_23_ply \
    --timesteps 0,10,20,30,40,50
```

**生成されるファイル**:
```
scene_23_ply/
├── frame_000.ply  # 時刻 t=0
├── frame_010.ply  # 時刻 t=10
├── frame_020.ply  # 時刻 t=20
...
└── frame_050.ply  # 時刻 t=50
```

#### タスク2: アニメーション確認

**方法1: 個別フレーム表示**
```bash
# 各時刻のPLYを個別に開く
meshlab scene_23_ply/frame_000.ply
meshlab scene_23_ply/frame_025.ply
meshlab scene_23_ply/frame_050.ply
```

**方法2: アニメーション動画生成**
```python
# scripts/render_ply_sequence.py
# 各PLYをレンダリング → 動画化
```

### 動的オブジェクトの確認方法

**車両の動き**:
- Frame 0とFrame 50のPLYを並べて比較
- 車両を表すGaussianクラスタの位置が変化

**歩行者の動き**:
- Deformable Gaussiansにより滑らかに変形
- 姿勢変化もGaussianの配置で表現

### 実装の詳細（時系列対応）

```python
import torch
from models.trainers.single import SingleTrainer

def export_timestep_ply(checkpoint_path, timestep, output_path):
    """特定の時刻tのPLYをエクスポート"""

    # モデル読み込み
    ckpt = torch.load(checkpoint_path)

    # Canonical Gaussians取得
    canonical_means = ckpt['gaussians']['Background']['means']
    canonical_quats = ckpt['gaussians']['Background']['quats']

    # Deformation Network適用
    t = torch.tensor([timestep / 50.0])  # 正規化された時刻
    deform_net = ckpt['deform_network']

    with torch.no_grad():
        # 変形計算
        delta_xyz, delta_rot, delta_scale = deform_net(
            canonical_means, t.repeat(len(canonical_means), 1)
        )

        # 変形後の位置
        deformed_means = canonical_means + delta_xyz
        deformed_quats = canonical_quats * delta_rot

    # PLYに保存
    save_ply(output_path, deformed_means, deformed_quats, ...)
```

### 使用例

```bash
# 時刻0, 25, 50のPLYをエクスポート
for t in 0 25 50; do
    python scripts/export_ply.py \
        --checkpoint logs/test_deformgs/first_test/scene_23_3cams/checkpoint_final.pth \
        --output scene_23_t${t}.ply \
        --timestep $t
done

# 並べて比較
meshlab scene_23_t0.ply scene_23_t25.ply scene_23_t50.ply
```

### 期待される結果

**静的要素（背景）**:
- 建物、道路：全フレームで同じ位置

**動的要素（車両・歩行者）**:
- フレーム間で位置が変化
- Gaussianクラスタが移動・変形

### 優先度

- **中**: 走行中シーン実験の後に実装
- **用途**: 3D構造の理解、デバッグ、他ツールとの連携

---

**最終更新**: 2026-02-14 22:20
**現在のステータス**: Scene 023完了、次は走行中シーン実験
**次のアクション**: シーンの移動距離分析 → Scene 114トレーニング
