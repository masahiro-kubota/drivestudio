# OmniRe の RTX 4090 実験パラメータ分析

## 📋 論文記載のパラメータ（原文引用）

### ハードウェア

**GPU**: NVIDIA RTX 4090

**原文**（Appendix A - Implementation Details）:
> "Our method runs on a single NVIDIA RTX 4090 GPU, with training for each scene taking about 1 hour."

**トレーニング時間**: 約1時間/シーン

---

### データセット

OmniReは**Waymoデータセット**をメインに使用し、追加で5つのデータセットで検証：

#### メインデータセット: Waymo Open Dataset

**原文**（Section 5 - Experiments）:
> "We conduct experiments on the Waymo Open Dataset (Sun et al., 2020), which comprises real-world driving logs. We tested up to 32 dynamic scenes in Waymo, including eight highly complex dynamic scenes that, in addition to typical vehicles, also contain diverse dynamic classes such as pedestrians and cyclists. Each selected segment contains approximately 150 frames."

**原文**（Baselines）:
> "For training, we utilize data from the three front-facing cameras, resized to a resolution of 640×960 for all methods, along with LiDAR data for supervision."

**パラメータ**:
```yaml
dataset: Waymo Open Dataset
scenes: 32 dynamic scenes (8 highly complex)
frames_per_scene: ~150
cameras: 3 (front-facing)
resolution: 640×960
background: 実世界（空マスクあり）
lidar: あり（初期化と深度監視に使用）
```

**特徴**:
- **マルチカメラ（3台）**: 前方3カメラ
- **実世界の都市シーン**: 車両、歩行者、自転車など多様な動的オブジェクト
- **LiDAR統合**: 深度監視とガウシアン初期化

#### 追加検証データセット

**原文**:
> "To further demonstrate our effectiveness on common driving scenes, we extend our results to 5 additional popular driving datasets: NuScenes (Caesar et al., 2020), Argoverse2 (Wilson et al., 2023), PandaSet (Xiao et al., 2021), KITTI (Geiger et al., 2012), and NuPlan (Caesar et al., 2021)."

```yaml
additional_datasets:
  - NuScenes
  - Argoverse2
  - PandaSet
  - KITTI
  - NuPlan
```

---

### トレーニングパラメータ

**原文**（Appendix A - Training）:
> "Our method trains for 30,000 iterations with all scene nodes optimized jointly."

**原文**（学習率設定）:
> "The learning rate for Gaussian properties aligns with the default settings of 3DGS (Kerbl et al., 2023), but varies slightly across different node types. Specifically, we set the learning rate for the rotation of Gaussians to 5×10⁻⁵ for non-rigid SMPL nodes and 1×10⁻⁵ for other nodes."

**原文**（球面調和関数の次数）:
> "The degrees of spherical harmonics are set to 3 for background, rigid nodes, and non-rigid deformable nodes, while it is set to 1 for non-rigid SMPL nodes."

```yaml
total_iterations: 30000

learning_rates:
  gaussian_rotation:
    smpl_nodes: 5e-5
    other_nodes: 1e-5
  instance_box_rotation: 1e-5 → 5e-6 (exponential decay)
  instance_box_translation: 5e-2 → 1e-5 (exponential decay)
  human_body_pose: 5e-5 → 1e-7 (exponential decay)

spherical_harmonics_degree:
  background: 3
  rigid_nodes: 3
  deformable_nodes: 3
  smpl_nodes: 1  # 歩行者のみ低次数

training_strategy: "すべてのシーンノードを同時最適化"
```

**重要**: OmniReは単一ステージでシーン全体を同時最適化（DeformableGSの2段階とは異なる）

---

### ネットワーク構造とシーングラフ

**原文**（Section 4.1 - Gaussian Scene Graph Modeling）:
> "Our scene graph is composed of the following nodes: (1) a Sky Node representing the sky that is far away from the ego-car, (2) a Background Node representing the static scene background such as buildings, roads, and vegetation, (3) a set of Rigid Nodes, each representing a rigidly movable object such as a vehicle, (4) a set of Non-rigid Nodes that include both pedestrians and cyclists."

```yaml
scene_graph_structure:
  sky_node:
    type: "Environment texture map"
    description: "遠方の空を表現"

  background_node:
    type: "Static Gaussians"
    initialization: "600k LiDAR + 400k random samples"
    total_points: 1000000

  rigid_nodes:
    type: "Static Gaussians + SE(3) transformation"
    objects: "車両（車、トラック）"
    description: "ローカル空間のガウシアンは時間で不変、剛体変換のみ"

  non_rigid_smpl_nodes:
    type: "Dynamic Gaussians + SMPL model"
    objects: "歩行者（走る・歩く）"
    control: "ジョイントレベル"
    template: "SMPL mesh (24 joints)"
    description: "LBS (Linear Blend Skinning) で変形"

  non_rigid_deformable_nodes:
    type: "Dynamic Gaussians + Deformation Network"
    objects: "自転車、遠方の歩行者、その他テンプレートなし動的物体"
    description: "共有変形フィールド + インスタンス埋め込み"
```

**変形ネットワーク**:

**原文**（Section 4.1 - Non-Rigid Deformable Nodes）:
> "we propose to use a general deformation network Fφ with parameter φ to learn the non-rigid motions within the nodes. [...] To maintain computational efficiency, the network weights share their identities of the nodes are disambiguated via an instance embedding parameter e_h."

```yaml
deformation_network:
  type: "共有MLP（全インスタンス共通）"
  input:
    - position_encoding(μ)  # ガウシアン位置
    - instance_embedding(e_h)  # インスタンス識別
    - time(t)
  output:
    - position_offset (δμ)
    - rotation_offset (δq)
    - scaling_offset (δs)

  weight_sharing: True  # 効率のため
  disambiguation: "インスタンス埋め込み e_h"
```

**重要な違い（DeformableGSとの比較）**:
- **DeformableGS**: シーン全体に単一の変形フィールド
- **OmniRe**: ノードごとに変形フィールド（重み共有、埋め込みで識別）

---

### 初期化

**原文**（Appendix A - Initialization）:
> "For the background model, we refer to PVG (Chen et al., 2023), combining 6×10⁵ LiDAR points with 4×10⁵ random samples, which are divided into 2×10⁵ near samples uniformly distributed by distance to the scene's origin and 2×10⁵ far samples uniformly distributed by inverse distance."

**原文**（LiDARフィルタリング）:
> "To initialize the background, we filter out the LiDAR samples of dynamic objects."

```yaml
initialization:
  background:
    lidar_points: 600000  # 動的オブジェクトを除外
    random_samples: 400000
      near_samples: 200000  # 距離で均一分布
      far_samples: 200000   # 逆距離で均一分布
    total: 1000000

  rigid_nodes:
    source: "累積LiDARポイント"

  non_rigid_deformable_nodes:
    source: "累積LiDARポイント"

  non_rigid_smpl_nodes:
    source: "SMPLテンプレートメッシュ上のガウシアン"
    pose: "カノニカル空間（Da pose）"
    method: "GART (Lei et al., 2023) と同様"

  color_initialization:
    visible_gaussians: "画像から投影"
    random_samples: "ランダムカラー"
```

---

### 適応的密度制御

**原文**（Appendix A - Training）:
> "For the Gaussian densification strategy, we utilize the absolute gradient Gaussians as introduced in Ye et al. (2024) to control memory usage. We set the densification threshold of position gradient to 3×10⁻⁴. [...] The densification threshold for scaling is 3×10⁻³."

```yaml
adaptive_density_control:
  strategy: "Absolute Gradient Gaussians (Ye et al., 2024)"
  purpose: "メモリ使用量の制御"

  thresholds:
    position_gradient: 3e-4  # 0.0003
    scaling: 3e-3            # 0.003

  operations:
    - pruning: "不透明度が低いガウシアンを削除"
    - cloning: "細かい幾何のためにクローン"
    - splitting: "大きく重なる領域を分割"

  note: "DeformableGS (2e-4) より高いしきい値"
```

**Absolute Gradient の効果**:
**原文**（Appendix D.4）:
> "This use of absolute gradient has a minimal impact on performance"

メモリ制御のため、性能への影響は最小限

---

### 損失関数

**原文**（Section 4.3 - Optimization）:
> "We use the following objective function for optimization: L = (1 - λ_s)L₁ + λ_s L_SSIM + λ_depth L_depth + λ_opacity L_opacity + L_reg"

**原文**（Appendix A - Optimization）:
> "We set the weight of the SSIM loss, λ_s, to 0.2, the depth loss, λ_depth, to 0.1, the opacity loss, λ_opacity, to 0.05, and the pose smoothness loss, λ_pose, to 0.01."

```yaml
loss_function:
  main_loss:
    L1_loss: "(1 - λ_s) × L1"
    SSIM_loss: "λ_s × L_SSIM"
    depth_loss: "λ_depth × L_depth"
    opacity_loss: "λ_opacity × L_opacity"
    regularization: "L_reg"

  weights:
    λ_s: 0.2        # SSIM
    λ_depth: 0.1    # 深度監視
    λ_opacity: 0.05 # 不透明度（空マスク）
    λ_pose: 0.01    # 人体ポーズ平滑化

  dynamic_region_weight: 5  # 動的領域の画像損失に高重み
```

**深度損失**:

**原文**（Equation 9）:
> "L_depth = 1/(hwc) Σ ||D^s - D̂||₁"
> "where D^s is the inverse of the sparse depth map. We project LiDAR points onto the image plane to generate the sparse LiDAR map"

```yaml
depth_supervision:
  source: "LiDAR sparse depth map"
  method: "L1 loss on inverse depth"
  projection: "LiDARポイントを画像平面に投影"
```

**ポーズ平滑化損失**:

**原文**（Equation 11）:
> "L_pose = 1/2 ||θ(t - δ) + θ(t + δ) - 2θ(t)||₁"
> "where δ is a randomly chosen integer from {1, 2, 3, 4, 5}"

```yaml
pose_smoothness:
  formula: "2次微分近似による平滑化"
  delta_range: [1, 2, 3, 4, 5]  # ランダム選択
  purpose: "時間的に滑らかな人体ポーズ"
```

---

### 人体ポーズ推定パイプライン

**原文**（Section 4.2 - Reconstructing In-the-Wild Humans）:
> "Reconstructing humans from driving logs faces challenges as in-the-wild pose estimators (Goel et al., 2023; Rajasegaran et al., 2022) are typically designed for single video input and often miss predictions in occlusion cases. We designed a pipeline that addresses these limitations to predict accurate and temporally consistent human poses from multi-view videos with frequent occlusions."

```yaml
human_pose_pipeline:
  step1_detection:
    method: "4D-Humans (Goel et al., 2023)"
    application: "各カメラの映像に独立適用"
    cameras: 3

  step2_tracklet_matching:
    input: "Ground truth tracklets + predicted tracklets"
    method: "Maximum mean IoU of 2D projections"
    output: "カメラごとのマッチング結果"

  step3_pose_completion:
    problem: "オクルージョンによる欠損"
    solution: "時間的補間"
    method: "Ground truth tracklets と比較して欠損を特定"

  step4_pose_refinement:
    method: "ガウシアンと同時最適化"
    loss: "同じ再構成損失"
    purpose: "予測誤差とスケール曖昧性の解消"
```

**重要**: 野外の厳しいオクルージョンに対応した独自パイプライン

---

### パフォーマンス

**レンダリングFPS**:

**原文**（Abstract）:
> "enabling advanced simulations (~60 Hz)"

```yaml
rendering_performance:
  fps: "~60 Hz (60 FPS)"
  gpu: RTX 4090
  note: "インタラクティブシミュレーション可能"
```

**トレーニング時間**:

**原文**（Appendix A）:
> "Our method runs on a single NVIDIA RTX 4090 GPU, with training for each scene taking about 1 hour. Training time varies with different training settings."

```yaml
training_performance:
  time: "約1時間/シーン"
  iterations: 30000
  gpu: RTX 4090 (single)
  variance: "設定により変動"
```

**定量的結果**:

**原文**（Section 5.1 - Main Results）:
> "The quantitative results in Tab. 1 show that OmniRe outperforms all other methods, with a significant margin in human-related regions, validating our holistic modeling of dynamic actors."

**原文**（Abstract）:
> "OmniRe achieves state-of-the-art performance [...] significantly outperforming previous methods in terms of full image metrics (+1.88 PSNR for reconstruction and +2.38 PSNR for NVS). The differences are pronounced for dynamic actors, such as vehicles (+1.18 PSNR), and humans (+4.09 PSNR for reconstruction and +3.06 PSNR for NVS)"

```yaml
performance_metrics:
  full_image:
    reconstruction: "+1.88 PSNR"
    novel_view_synthesis: "+2.38 PSNR"

  vehicles:
    improvement: "+1.18 PSNR"

  humans:
    reconstruction: "+4.09 PSNR"
    novel_view_synthesis: "+3.06 PSNR"

  note: "特に人間領域で大幅改善"
```

---

## 🔍 DriveStudio の Waymo 設定との比較

### 現在の DriveStudio 設定（3cams.yaml）

```yaml
# DriveStudio (OmniRe ベース)
dataset: waymo
cameras: [0, 1, 2]  # 3台のカメラ
original_resolution:
  front_camera: [1920, 1280]
  front_left_camera: [1920, 1280]
  front_right_camera: [1920, 1280]

downscale_when_loading: [2, 2, 2]
# → 実際のロード解像度: [960, 640]

downscale: 1  # さらなるダウンスケールなし
# → 最終解像度: [960, 640]

load_lidar: True
load_sky_mask: True
load_dynamic_mask: True
load_objects: True  # バウンディングボックス
load_smpl: True     # 歩行者用SMPLテンプレート
```

---

### 重要な違い

| パラメータ | OmniRe (論文) | DriveStudio (現在) |
|-----------|---------------|-------------------|
| **データセット** | Waymo Open Dataset | Waymo |
| **カメラ数** | 3（前方3カメラ） | 3（前方3カメラ） |
| **解像度** | 640×960 | 960×640（**逆**） |
| **GPU** | RTX 4090 | RTX 3090（想定） |
| **入力タイプ** | 画像 + LiDAR | 画像 + LiDAR |
| **イテレーション** | 30,000 | 不明 |
| **動的オブジェクト** | シーングラフ（個別） | シーングラフ（個別） |
| **SMPLモデル** | あり（歩行者用） | あり（歩行者用） |
| **背景初期化** | 1,000,000 ポイント | 不明 |
| **SH次数** | 3（背景/車両）、1（SMPL） | 不明 |
| **アプローチ** | Gaussian Scene Graphs | Gaussian Scene Graphs |

**注意**: DriveStudioの解像度は **960×640** だが、OmniRe論文では **640×960**（幅×高さが逆）

---

## 📊 OmniRe を Waymo で動かす場合の推奨パラメータ

### シナリオ1: 論文に忠実な実装（RTX 4090）

```yaml
# OmniRe 論文設定の再現
data:
  dataset: waymo
  cameras: [0, 1, 2]  # 前方3カメラ
  resolution: 640×960  # 論文通り
  frames_per_scene: 150
  load_lidar: True
  load_sky_mask: True
  load_dynamic_mask: True
  load_objects: True  # バウンディングボックス
  load_smpl: True

training:
  total_iterations: 30000
  optimizer: Adam

  learning_rate:
    gaussian_rotation_smpl: 5e-5
    gaussian_rotation_others: 1e-5
    box_rotation: 1e-5 → 5e-6
    box_translation: 5e-2 → 1e-5
    human_pose: 5e-5 → 1e-7

initialization:
  background_lidar: 600000
  background_random: 400000
  total_background: 1000000

network:
  scene_graph:
    - sky_node: environment_map
    - background_node: static_gaussians
    - rigid_nodes: vehicles
    - smpl_nodes: pedestrians (with SMPL)
    - deformable_nodes: cyclists, others

  spherical_harmonics:
    background: 3
    rigid: 3
    deformable: 3
    smpl: 1

density_control:
  method: "Absolute Gradient"
  position_threshold: 3e-4
  scaling_threshold: 3e-3

loss:
  λ_ssim: 0.2
  λ_depth: 0.1
  λ_opacity: 0.05
  λ_pose: 0.01
  dynamic_region_weight: 5

human_pose:
  detector: "4D-Humans"
  pipeline:
    - tracklet_matching
    - pose_completion
    - pose_refinement
```

**推定VRAM**: 3-5 GB（100万背景 + 複数動的ノード）

**トレーニング時間**: 約1時間/シーン（RTX 4090）

---

### シナリオ2: RTX 3090 向けメモリ最適化版

```yaml
# メモリ制約を考慮した設定
data:
  cameras: [0, 1, 2]  # 3カメラ維持
  downscale_when_loading: [2, 2, 2]
  # → 960×640 (DriveStudio現在の設定)
  load_lidar: True
  lidar_downsample_factor: 4  # LiDARをダウンサンプル

initialization:
  background_lidar: 400000  # 600k → 400k
  background_random: 300000  # 400k → 300k
  total_background: 700000   # 1M → 700k

density_control:
  method: "Absolute Gradient"
  position_threshold: 4e-4  # より高いしきい値
  scaling_threshold: 4e-3
  max_gaussians: 800000  # ガウシアン数を制限

training:
  total_iterations: 30000  # 維持
  # その他のパラメータは同じ

# SH次数は維持（メモリ影響小）
spherical_harmonics:
  background: 3
  rigid: 3
  deformable: 3
  smpl: 1
```

**推定VRAM**: 2-3 GB（積極的なダウンサンプリング）

**トレーニング時間**: 1.5-2時間/シーン（RTX 3090、論文より遅い）

**注意**:
- RTX 4090 → RTX 3090 では性能低下の可能性
- Absolute Gradientのしきい値を上げてメモリを節約
- 背景ポイント数を削減（1M → 700k）

---

### シナリオ3: 単眼カメラ版（最小メモリ）

```yaml
# 最もシンプルな設定（DeformableGS風）
data:
  cameras: [0]  # 前方カメラのみ
  resolution: 640×960
  load_lidar: True
  load_sky_mask: True
  load_dynamic_mask: True
  load_objects: True
  load_smpl: True  # SMPLは維持

initialization:
  background_lidar: 300000
  background_random: 200000
  total_background: 500000

density_control:
  position_threshold: 5e-4
  max_gaussians: 500000

training:
  total_iterations: 30000
```

**推定VRAM**: 1-2 GB

**トレードオフ**: マルチビュー情報が減少、品質低下の可能性

---

## ❓ よくある質問

### Q1: OmniRe は DriveStudio のベースですか？

**A**: はい、**DriveStudioはOmniReの公式実装です**：

**README.mdより**:
> "This codebase also contains the **official implementation** of: OmniRe: Omni Urban Scene Reconstruction"

DriveStudioは：
- ✅ OmniReの**公式実装コードベース**
- ✅ OmniRe論文の実験を再現可能
- ✅ 加えて、Deformable-GS、PVG、Street Gaussiansの非公式実装も含む

### Q2: DeformableGS と OmniRe の主な違いは？

**A**: **アプローチの根本的な違い**：

| 特徴 | DeformableGS | OmniRe |
|-----|-------------|--------|
| **変形モデル** | シーン全体に単一フィールド | ノードごとに変形フィールド |
| **カメラ** | 単眼（1カメラ） | マルチカメラ（3カメラ） |
| **動的物体** | 暗黙的（単一フィールド） | 明示的（シーングラフ） |
| **人間モデル** | なし | SMPL（ジョイントレベル制御） |
| **適用シーン** | シンプルな動的シーン | 複雑な都市シーン |
| **制御性** | なし | 高い（個別編集可能） |
| **GPU** | RTX 3090 | RTX 4090 |

**OmniReの論文からの引用**（Related Work）:
> "Yang et al. (2023c) (DeformableGS): utilizes a single deformation network for the entire scene, and usually fail in highly complex scenes with many movements."

### Q3: RTX 3090 で OmniRe は動きますか？

**A**: **動くが制限あり**：

✅ **可能な部分**:
- メモリ最適化版の実装
- シンプルなシーン（動的物体が少ない）
- Absolute Gradientによるメモリ制御

⚠️ **困難な部分**:
- 複雑なシーン（多数の歩行者+車両）
- 論文通りの100万背景ポイント
- トレーニング時間の増加

**推奨**:
- **RTX 3090**: シンプル〜中程度のシーン
- **RTX 4090**: 複雑なシーン（論文推奨）

論文はRTX 4090を使用しており、RTX 3090での性能保証はありません。

### Q4: 人体ポーズ推定は必須ですか？

**A**: **歩行者を含むシーンでは必須**：

必要な場合:
- ✅ 歩行者がシーンに存在
- ✅ ジョイントレベルの制御が必要
- ✅ 人間中心のシミュレーション

不要な場合:
- ❌ 車両のみのシーン
- ❌ 遠方の歩行者のみ（Deformable Nodesで対応可能）

**原文**（Section 4.1 - Non-Rigid SMPL Nodes）:
> "it is highly challenging to accurately optimize the SMPL poses θ(t) from scratch just based on sparse observations. Hence a rough initialization of θ(t) is typically needed"

→ **4D-Humans**による初期化が事実上必須

### Q5: DriveStudioの解像度設定（960×640）は正しいですか？

**A**: **論文と逆ですが、問題ない可能性**：

- **OmniRe論文**: 640×960（幅×高さ）
- **DriveStudio**: 960×640（幅×高さ）

考えられる理由:
1. **表記の違い**: 論文は (高さ×幅)、DriveStudioは (幅×高さ)
2. **カメラの向き**: 縦長 vs 横長の違い
3. **意図的な変更**: Waymoカメラの実際のアスペクト比に合わせた

確認方法:
```bash
# Waymoカメラの実際の解像度を確認
# original_resolution: [1920, 1280] → アスペクト比 3:2 (横長)
```

→ **960×640 (3:2)** が正しい可能性が高い

---

## 📝 まとめ

### OmniRe (RTX 4090) の実験設定

```yaml
GPU: RTX 4090
Dataset: Waymo Open Dataset (32 scenes)
Resolution: 640×960
Cameras: 3 (front-facing)
Iterations: 30,000
Background Init: 1,000,000 points (600k LiDAR + 400k random)
Scene Graph: Sky + Background + Rigid + SMPL + Deformable
SH Degrees: 3 (background/vehicles), 1 (SMPL pedestrians)
Densification: Absolute Gradient (3×10⁻⁴)
Training Time: ~1 hour/scene
Rendering: ~60 FPS
VRAM: 3-5 GB (推定)
```

### Waymo（DriveStudio）で使う場合

**オプション1: 論文に忠実な実装（RTX 4090）**
- 3カメラ、640×960、100万背景ポイント
- 複雑な都市シーンに対応
- 人間中心のシミュレーション可能

**オプション2: RTX 3090 最適化版**
- 3カメラ維持、960×640
- 背景ポイント削減（700k）
- Absolute Gradient しきい値調整

**オプション3: 単眼最小版**
- 1カメラ、メモリ最小化
- シンプルなシーンのみ

### DeformableGS との比較

| 特徴 | DeformableGS | OmniRe |
|-----|-------------|--------|
| **GPU** | RTX 3090 | RTX 4090 |
| **カメラ** | 1（単眼） | 3（マルチ） |
| **解像度** | 800×800 | 640×960 |
| **イテレーション** | 40,000 (3k warmup) | 30,000 |
| **背景初期化** | 不明 | 1,000,000 |
| **変形フィールド** | シーン全体に1つ | ノードごと |
| **人間モデル** | なし | SMPL |
| **適用** | シンプルなシーン | 複雑な都市シーン |
| **VRAM推定** | 1-2 GB | 3-5 GB |

**推奨**:
- **シンプルなシーン**: DeformableGS (RTX 3090で可能)
- **複雑な都市シーン**: OmniRe (RTX 4090推奨、RTX 3090で制限あり)
- **DriveStudio**: OmniReベース（マルチカメラ、SMPL、シーングラフ）
