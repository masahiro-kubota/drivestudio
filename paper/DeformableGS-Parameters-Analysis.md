# DeformableGS の RTX 3090 実験パラメータ分析

## 📋 論文記載のパラメータ（原文引用）

### ハードウェア

**GPU**: NVIDIA RTX 3090 (24GB VRAM)

**原文**:
> "All the experiments were done on an NVIDIA RTX 3090."

---

### データセット

DeformableGSは**3種類のデータセット**で実験を実施：

#### 1. 合成データセット（D-NeRF）

**原文**:
> "Experiments with synthetic datasets were all conducted against a black background and at a full resolution of 800x800."

**パラメータ**:
```yaml
dataset: D-NeRF synthetic dataset
resolution: 800×800
background: black
camera_type: monocular (単眼)
```

**特徴**:
- **単眼動的シーン**
- 合成データなので完全にコントロールされた環境
- フル解像度で処理

#### 2. 実世界データセット（NeRF-DS）

**原文**:
> "We compare our method with the baselines using the monocular real-world dataset from NeRF-DS [30]"

**パラメータ**:
```yaml
dataset: NeRF-DS
type: real-world monocular dataset
resolution: 論文に明記なし（NeRF-DS論文の設定に従う）
```

#### 3. 実世界データセット（HyperNeRF）

**原文**:
> "We compare our method with the baselines using the monocular real-world dataset from [...] HyperNeRF [31]. It should be noted that the camera poses for some of the HyperNeRF datasets are very inaccurate."

**パラメータ**:
```yaml
dataset: HyperNeRF
type: real-world monocular dataset
note: "カメラポーズが非常に不正確なものあり"
```

**重要**: HyperNeRFは定量評価からは除外（ポーズ不正確のため）

---

### トレーニングパラメータ

**原文**:
> "For training, we conducted training for a total of 40k iterations. During the initial 3k iterations, we solely trained the 3D Gaussians to attain relatively stable positions and shapes. Subsequently, we jointly train the 3D Gaussians and the deformation field."

```yaml
total_iterations: 40000
phase1_iterations: 3000  # 3D Gaussiansのみ
phase2_iterations: 37000  # 共同トレーニング

optimizer: Adam
beta_values: [0.9, 0.999]

learning_rates:
  gaussian_3d: "3D-GS公式実装と同じ"
  deformation_network:
    initial: 8e-4
    final: 1.6e-6
    decay: exponential
```

---

### ネットワーク構造

**変形フィールドMLP**:

**原文**（Section 3.2）:
> "We set the depth of the deformation network D = 8 and the dimension of the hidden layer W = 256."

**原文**（Appendix B.1）:
> "our MLP Fθ initially processes the input through eight fully connected layers that include ReLU activations and feature 256-dimensional hidden layers"

```yaml
deformation_network:
  depth: 8
  hidden_dim: 256
  activation: ReLU
  input:
    - position_encoding(x)  # L=10 (synthetic), L=10 (real)
    - position_encoding(t)  # L=6 (synthetic), L=10 (real)
  output:
    - position_offset (δx)
    - rotation_offset (δr)
    - scaling_offset (δs)

  storage: 2MB  # 追加ストレージ
```

**位置エンコーディング**:

**原文**（Section 3.2）:
> "where L = 10 for x and L = 6 for t in synthetic scenes, while L = 10 for both x and t in real scenes."

```yaml
positional_encoding:
  synthetic_scenes:
    position_x: L=10
    time_t: L=6
  real_scenes:
    position_x: L=10
    time_t: L=10
```

---

### 適応的密度制御

**原文**（Section 3.1）:
> "Following [15], we discern the 3D Gaussians that demand adjustments using a threshold set by t_pos = 0.0002. For diminutive Gaussians inadequate for capturing geometric details, we clone the Gaussians and move them in a certain distance in the direction of the positional gradients. Conversely, for those that are conspicuously large and overlapping, we split them and divide their scale by a hyperparameter ξ = 1.6."

```yaml
adaptive_density_control:
  position_gradient_threshold: 0.0002
  split_scale_factor: 1.6

  operations:
    - pruning: "不透明度が低いガウシアンを削除"
    - cloning: "細かい幾何のためにクローン+移動"
    - splitting: "大きく重なる領域を分割"
```

---

### 損失関数

**原文**（Appendix B.2）:
> "We then jointly optimize both the deformation network and the 3D Gaussians using a combination of L₁ loss and D-SSIM loss"
> "where λ = 0.2 is used in all our experiments."

```yaml
loss:
  type: "L1 + D-SSIM"
  formula: "(1 - λ)L₁ + λ L_D-SSIM"
  lambda: 0.2
```

---

### Annealing Smooth Training (AST)

**原文**（Section 3.3、論文記載の式より）:

```yaml
annealing_smooth_training:
  noise: "Gaussian N(0,1)"
  scaling_factor_beta: 0.1
  threshold_iteration_tau: 20000  # 20k
  formula: "X(i) = N(0,1) · β · Δt · (1 - i/τ)"

  effect:
    - "初期段階: 時間的汎化を向上"
    - "後期段階: ディテールを保持"
    - "追加オーバーヘッド: なし"
```

---

### パフォーマンス

**ガウシアン数とFPS**:

**原文**（Section 4.2 - Rendering Efficiency）:
> "when the number of 3D Gaussians is below 250k, our method can achieve real-time rendering over 30 FPS on an NVIDIA RTX 3090."

```yaml
performance:
  target_gaussians: "< 250,000"
  rendering_fps: "> 30 FPS"
  gpu: RTX 3090

  note: "250k を超えるとメモリとパフォーマンスが低下"
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

| パラメータ | DeformableGS (論文) | DriveStudio (現在) |
|-----------|---------------------|-------------------|
| **データセット** | D-NeRF, NeRF-DS, HyperNeRF | Waymo |
| **カメラ数** | 1（単眼） | 3（マルチカメラ） |
| **解像度** | 800×800 | 960×640（ダウンスケール後） |
| **入力タイプ** | 画像のみ | 画像 + LiDAR |
| **動的オブジェクト** | シーン全体を変形 | 個別にモデル化（OmniRe） |
| **背景** | 黒背景 | 実世界（空マスクあり） |
| **SMPLモデル** | なし | あり（歩行者用） |
| **アプローチ** | 単一変形フィールド | シーングラフ（複数ノード） |

---

## 📊 DeformableGS を Waymo で動かす場合の推奨パラメータ

### シナリオ1: 論文に忠実な実装（単眼、単一シーン）

```yaml
# 最もシンプルな設定
data:
  dataset: waymo
  cameras: [0]  # 前方カメラのみ
  resolution: 800×800  # 論文と同じ
  background: black
  load_lidar: False
  load_objects: False
  load_smpl: False

training:
  total_iterations: 40000
  warmup_iterations: 3000
  optimizer: Adam
  beta: [0.9, 0.999]

  learning_rate:
    deformation_network:
      initial: 8e-4
      final: 1.6e-6
      schedule: exponential

network:
  deformation_mlp:
    depth: 8
    hidden_dim: 256
    positional_encoding:
      position: L=10
      time: L=10

density_control:
  threshold: 0.0002
  split_scale: 1.6

loss:
  lambda_ssim: 0.2

annealing_smooth_training:
  enabled: True
  beta: 0.1
  tau: 20000
```

**推定VRAM**: 1-2 GB（250k ガウシアン）

---

### シナリオ2: Waymo に適応（マルチカメラ、LiDAR活用）

```yaml
# より実用的な設定
data:
  dataset: waymo
  cameras: [0, 1, 2]  # 3カメラ
  downscale_when_loading: [2, 2, 2]
  # → 960×640
  load_lidar: True  # LiDARで初期化とAABB計算
  load_sky_mask: True
  load_dynamic_mask: True

training:
  total_iterations: 40000
  warmup_iterations: 3000
  # その他は同じ

# DeformableGS は単一変形フィールド
# → Waymo の複雑な動的シーンでは限界あり
```

**推定VRAM**: 2-3 GB（より多くのカメラとポイント）

**注意**:
- DeformableGSは**シーン全体に単一の変形フィールド**を使用
- Waymo のような**複数の動的オブジェクト**がある場合は苦戦する可能性
- OmniReの論文でも指摘:
  > "単一の変形ネットワークはシーン全体に → 複雑なシーンで失敗"

---

### シナリオ3: RTX 3090でメモリ制約を考慮

```yaml
# メモリ最適化版
data:
  cameras: [0]  # 単眼に制限
  downscale_when_loading: [4]  # より積極的なダウンスケール
  # → 480×320
  load_lidar: True
  lidar_downsample_factor: 8  # LiDARもダウンサンプル

training:
  max_gaussians: 200000  # 250k未満に制限

density_control:
  threshold: 0.0005  # より高いしきい値でガウシアン数を抑制
  prune_frequency: 100  # 頻繁にプルーニング

# Absolute Gradient推奨（OmniRe の知見より）
use_absgrad: True
```

**推定VRAM**: 1 GB以下（保守的な設定）

---

## ❓ よくある質問

### Q1: DeformableGS は Waymo データセットで動きますか？

**A**: 技術的には可能ですが、**制限があります**：

✅ **動く部分**:
- 単眼カメラ（front_camera のみ使用）
- 比較的シンプルな動的シーン
- RTX 3090で十分なメモリ

⚠️ **困難な部分**:
- 複数の独立した動的オブジェクト（車両、歩行者など）
- 単一の変形フィールドでは表現力が不足
- OmniReのように個別モデリングが必要

**OmniRe論文の指摘**:
> "Yang et al. (2023c)（DeformableGS）: シーン全体に単一の変形ネットワーク → 多くの動きを含む複雑なシーンで失敗"

### Q2: 現在のDriveStudioはOmniReベースですか？

**A**: はい、**DriveStudioはOmniReの公式実装です**：

**README.mdより**:
> "This codebase also contains the **official implementation** of: OmniRe: Omni Urban Scene Reconstruction"

DriveStudioは：
- OmniReの公式実装コードベース
- 加えて、Deformable-GS、PVG、Street Gaussiansの非公式実装も含む

DeformableGSとの違い:
- DeformableGS: シーン全体に単一変形フィールド
- OmniRe（DriveStudio）: オブジェクトごとにモデリング（シーングラフ）

### Q3: RTX 3090で十分ですか？

**A**: **シーンの複雑さ次第**：

✅ **DeformableGS風の単純なシーン**:
- 単眼カメラ
- 250k未満のガウシアン
- → RTX 3090で快適（1-2 GB VRAM）

⚠️ **OmniRe風の複雑なシーン**:
- 3カメラ
- 背景100万 + 複数の動的オブジェクト
- → RTX 3090でギリギリ（3-4 GB VRAM推定）
- → **Absolute Gradient必須**

🎯 **推奨**:
- シンプルなシーン: RTX 3090で十分
- 複雑なシーン: RTX 4090推奨（OmniRe論文使用）

---

## 📝 まとめ

### DeformableGS (RTX 3090) の実験設定

```yaml
GPU: RTX 3090 (24GB)
Dataset: D-NeRF (synthetic)
Resolution: 800×800
Camera: monocular (単眼)
Iterations: 40,000 (warmup 3,000)
Network: MLP 8層×256次元
Storage: +2MB
Max Gaussians: ~250,000
Performance: 30+ FPS
VRAM: 1-2 GB (推定)
```

### Waymo（DriveStudio）で使う場合

**オプション1: 忠実な実装**
- 単眼、800×800、論文と同じ設定
- シンプルなシーンのみ対応

**オプション2: 実用的な実装**
- マルチカメラ、LiDAR活用
- ただし複雑なシーンでは限界あり
- → **OmniReのような個別モデリングが必要**

**推奨**:
Waymoのような複雑な都市シーンには、**OmniReアプローチ**（現在のDriveStudio）の方が適している可能性が高い
