# 3D Gaussian Splatting トレーニング時のVRAM使用量分析（原文引用付き）

> **注意**: このドキュメントでは、論文からの直接引用（✅確認済み）と推測（⚠️推測）を明確に区別しています。

---

## 📋 目次

1. [論文に明記されている事実](#1-論文に明記されている事実)
2. [推測に基づく計算](#2-推測に基づく計算)
3. [実用的な推奨事項](#3-実用的な推奨事項)

---

## 1. 論文に明記されている事実

### 1.1 Deformable 3D Gaussians

#### ✅ GPU仕様

**セクション**: 4.1. Implementation Details

**原文**:
> "All the experiments were done on an NVIDIA RTX 3090."

**補足**: RTX 3090は24GB VRAMを搭載（一般的な仕様）

---

#### ✅ ガウシアン数とレンダリング性能

**セクション**: 4.2. Results and Comparisons - Rendering Efficiency

**原文**:
> "The rendering speed is correlated with the quantity of 3D Gaussians. Overall, when the number of 3D Gaussians is below 250k, our method can achieve real-time rendering over 30 FPS on an NVIDIA RTX 3090."

**セクション**: C.3. Results on Rendering Efficiency (Appendix)

**原文**:
> "In our research, we present comprehensive Frames Per Second (FPS) testing results in Tab. 4. Tests were conducted on one NVIDIA RTX 3090. It is observed that when the number of point clouds remains below ~250k, our method can achieve real-time rendering at rates greater than 30 FPS."

**要点**:
- **250,000個以下**のガウシアンで30 FPS以上を達成
- ガウシアン数とレンダリング速度は相関関係

---

#### ✅ 変形ネットワークのストレージサイズ

**セクション**: B.1. Network Architecture of the Deformation Field (Appendix)

**原文**:
> "Due to the compact structure of MLP, our additional storage compared to 3D Gaussians is only 2MB."

**要点**: 変形フィールドMLPは3D Gaussiansに対して**わずか2MB**の追加ストレージ

---

#### ✅ ネットワーク構造

**セクション**: 3.2. Deformable 3D Gaussians

**原文**:
> "We set the depth of the deformation network D = 8 and the dimension of the hidden layer W = 256."

**セクション**: B.1. Network Architecture of the Deformation Field (Appendix)

**原文**:
> "As shown in Fig. 7, our MLP Fθ initially processes the input through eight fully connected layers that include ReLU activations and feature 256-dimensional hidden layers, and outputs a 256-dimensional feature vector. This feature vector is subsequently passed through three additional fully connected layers (without activation) to separately output the offsets over time for position, rotation, and scaling."

**要点**:
- **8層**の全結合層
- 隠れ層: **256次元**
- 追加の3層で位置、回転、スケーリングのオフセットを出力

---

#### ✅ 画像解像度

**セクション**: 4.1. Implementation Details

**原文**:
> "Experiments with synthetic datasets were all conducted against a black background and at a full resolution of 800x800."

**要点**: 合成データセットは**800×800**の解像度

---

#### ✅ トレーニング設定

**セクション**: 4.1. Implementation Details

**原文**:
> "For training, we conducted training for a total of 40k iterations. During the initial 3k iterations, we solely trained the 3D Gaussians to attain relatively stable positions and shapes. Subsequently, we jointly train the 3D Gaussians and the deformation field."

**セクション**: 3.1. Differentiable Rendering (Method)

**原文**:
> "Experimental results show that after 30k training iterations, the shape of the 3D Gaussians stabilizes, as does the canonical space, which indirectly proves the efficacy of our design."

**要点**:
- 総イテレーション: **40,000回**
- 初期3,000回: 3D Gaussiansのみトレーニング
- その後37,000回: 共同トレーニング
- 30,000回後に形状が安定化

---

#### ✅ メモリ消費に関する制限事項

**セクション**: 5. Limitations

**原文**:
> "Furthermore, the temporal complexity of our approach is directly proportional to the quantity of 3D Gaussians. In scenarios with an extensive array of 3D Gaussians, there is a potential escalation in both training duration and memory consumption."

**要点**:
- ガウシアン数に**比例**して時間的複雑性が増加
- 大量のガウシアンでは**トレーニング時間とメモリ消費が増加**する可能性

---

#### ✅ 位置エンコーディング

**セクション**: 3.2. Deformable 3D Gaussians

**原文**:
> "where L = 10 for x and L = 6 for t in synthetic scenes, while L = 10 for both x and t in real scenes."

**要点**:
- 合成シーン: L=10 (位置x), L=6 (時間t)
- 実シーン: L=10 (位置x, 時間t両方)

---

#### ❌ 球面調和関数（SH）の次数

**記載なし**: 論文内で具体的な次数は明記されていません

---

#### ❌ VRAM使用量の具体的な数値

**記載なし**: トレーニング時やレンダリング時のVRAM使用量の具体的な数値は記載されていません

---

### 1.2 OmniRe

#### ✅ GPU仕様とトレーニング時間

**セクション**: A. IMPLEMENTATION DETAILS (Appendix)

**原文**:
> "Our method runs on a single NVIDIA RTX 4090 GPU, with training for each scene taking about 1 hour. Training time varies with different training settings."

**要点**:
- GPU: **NVIDIA RTX 4090**（24GB VRAM搭載）
- トレーニング時間: **約1時間/シーン**
- 設定により変動

---

#### ✅ 背景ガウシアンの初期化

**セクション**: A. IMPLEMENTATION DETAILS - Initialization (Appendix)

**原文**:
> "For the background model, we refer to PVG (Chen et al., 2023), combining 6 × 10⁵ LiDAR points with 4 × 10⁵ random samples, which are divided into 2 × 10⁵ near samples uniformly distributed by distance to the scene's origin and 2 × 10⁵ far samples uniformly distributed by inverse distance."

**要点**:
- LiDARポイント: **600,000個**
- ランダムサンプル: **400,000個**
  - 近距離: 200,000個
  - 遠距離: 200,000個
- **合計: 1,000,000個**の初期ポイント

---

#### ✅ 画像解像度

**セクション**: 5. EXPERIMENTS - Baselines

**原文**:
> "For training, we utilize data from the three front-facing cameras, resized to a resolution of 640×960 for all methods, along with LiDAR data for supervision."

**要点**:
- 解像度: **640×960**
- カメラ: **3台**（前方）
- すべての手法で統一

---

#### ✅ 球面調和関数（SH）の次数

**セクション**: A. IMPLEMENTATION DETAILS - Training (Appendix)

**原文**:
> "The degrees of spherical harmonics are set to 3 for background, rigid nodes, and non-rigid deformable nodes, while it is set to 1 for non-rigid SMPL nodes."

**要点**:
- 背景ノード: **次数3**
- 剛体ノード（車両）: **次数3**
- 非剛体変形可能ノード: **次数3**
- 非剛体SMPLノード（歩行者）: **次数1**

**重要**: SMPLノードで次数を下げることでメモリ効率化

---

#### ✅ トレーニング設定

**セクション**: A. IMPLEMENTATION DETAILS - Training (Appendix)

**原文**:
> "Our method trains for 30,000 iterations with all scene nodes optimized jointly."

**要点**: **30,000イテレーション**で全ノードを共同最適化

---

#### ✅ メモリ制御戦略

**セクション**: A. IMPLEMENTATION DETAILS - Training (Appendix)

**原文**:
> "For the Gaussian densification strategy, we utilize the absolute gradient Gaussians as introduced in Ye et al. (2024) to control memory usage. We set the densification threshold of position gradient to 3 × 10⁻⁴. This use of absolute gradient has a minimal impact on performance, as shown in Appendix D.4. The densification threshold for scaling is 3 × 10⁻³."

**要点**:
- **Absolute Gradient Gaussians**を使用して**メモリ使用量を制御**
- 位置勾配の高密度化しきい値: **3×10⁻⁴**
- スケーリングの高密度化しきい値: **3×10⁻³**
- 性能への影響は最小限

---

#### ✅ Absolute Gradientの重要性（アブレーション研究）

**セクション**: D.4. ABLATION STUDIES - Absolute Gradient (Appendix)

**原文**（Table 11の説明文より）:
> "We observe that 1) Disabling AbsGrad leads to a marginal performance decrease (about 0.1 PSNR) for all methods, proving that AbsGrad is not the key factor contributing to our performance advantage over others. Note that DeformableGS fails to run due to out-of-memory issues when AbsGrad is disabled."

**要点**:
- AbsGradなしでは約0.1 PSNR低下
- **DeformableGSはAbsGradなしでOOM（メモリ不足）**
- AbsGradは性能向上よりもメモリ安定化に重要

---

#### ❌ ネットワークサイズの詳細

**記載なし**: 変形ネットワーク Fφ の具体的な構造やパラメータ数は明記されていません

---

#### ❌ VRAM使用量の具体的な数値

**記載なし**: トレーニング時やレンダリング時のVRAM使用量の具体的な数値は記載されていません

---

## 2. 推測に基づく計算

> ⚠️ **警告**: 以下の計算は論文に明記されていない情報を含みます。一般的な3DGSの実装に基づく推測です。

### 2.1 各ガウシアンのメモリサイズ（推測）

#### 計算根拠

3D Gaussianの一般的なパラメータ構成（3DGS論文 Kerbl et al., 2023より）:

```
パラメータ:
- 位置 μ (3D座標):        3 × 4 bytes = 12 bytes
- 不透明度 o:             1 × 4 bytes = 4 bytes
- 回転 q (四元数):        4 × 4 bytes = 16 bytes
- スケーリング s:         3 × 4 bytes = 12 bytes
- SH係数 c:               計算が必要
```

#### 球面調和関数（SH）係数のサイズ

SH次数nに対する係数数: (n+1)²

| 次数 | 係数数 | RGB 3ch | メモリ |
|-----|-------|---------|--------|
| 0   | 1     | 3       | 12 B   |
| 1   | 4     | 12      | 48 B   |
| 2   | 9     | 27      | 108 B  |
| 3   | 16    | 48      | 192 B  |

#### ⚠️ 推定: 各ガウシアンのメモリ（SH次数3の場合）

```
合計 = 12 + 4 + 16 + 12 + 192 = 236 bytes/ガウシアン
```

---

### 2.2 トレーニング時のメモリ倍率（推測）

**Adam Optimizer**を使用する場合（両論文で使用）:
- パラメータ本体: 1×
- 勾配: 1×
- 1次モーメント: 1×
- 2次モーメント: 1×
- **合計: 4倍**（最低限）

**中間バッファ**（ラスタライゼーション、ソートなど）:
- 推定: 2-5倍

**⚠️ 推定総倍率: 5-10倍**

---

### 2.3 VRAM使用量の推定計算

#### Deformable 3D Gaussians（250k ガウシアン、SH次数3と仮定）

```
ガウシアン本体: 250,000 × 236 bytes = 59 MB
トレーニング時: 59 MB × 5-10倍 = 295-590 MB
変形ネットワーク: 2 MB（論文記載）
その他バッファ: 推定 200-400 MB

⚠️ 推定総VRAM: 約 500 MB - 1 GB
```

**注意**: 実際のVRAM使用量は1-2 GBと推測されますが、論文に明記はありません。

#### OmniRe（1,000k 背景 + 動的オブジェクト）

**背景のみ**:
```
ガウシアン本体: 1,000,000 × 236 bytes = 236 MB
（SH次数3と仮定）
```

**SMPLノード**（SH次数1）:
```
1ガウシアンあたり: 12 + 4 + 16 + 12 + 48 = 92 bytes
```

**⚠️ 複雑なシーンの推定例**:
```
背景（SH次数3）:       1,000,000 × 236 B = 236 MB
車両10台（SH次数3）:     300,000 × 236 B = 71 MB
歩行者5人（SH次数1）:     50,000 × 92 B = 4.6 MB
その他（SH次数3）:        30,000 × 236 B = 7 MB
─────────────────────────────────────────
合計ガウシアン本体:                    約 319 MB

トレーニング時（8-12倍と推定）:        2.5-3.8 GB
変形ネットワーク（推定）:              5-10 MB
その他バッファ:                        500 MB - 1 GB
─────────────────────────────────────────
⚠️ 推定総VRAM:                        3-5 GB
```

**重要な注意**:
1. これらは推測であり、論文に明記されていません
2. 実際のVRAM使用量は実装詳細により大きく異なる可能性があります
3. SH次数が明記されていない部分（Deformableなど）は仮定に基づきます

---

## 3. 論文記載と推測の対照表

| 項目 | Deformable 3D GS | OmniRe | 出典 |
|------|------------------|--------|------|
| GPU | RTX 3090 | RTX 4090 | ✅ 論文記載 |
| VRAM | 24GB（推定） | 24GB（推定） | ⚠️ 一般仕様 |
| ガウシアン数 | ~250k（最適） | 1,000k（背景） | ✅ 論文記載 |
| SH次数 | **記載なし** | 3 or 1（ノードにより） | ❌/✅ |
| 解像度 | 800×800 | 640×960 | ✅ 論文記載 |
| ネットワーク | 8層、256次元 | **記載なし** | ✅/❌ |
| 追加ストレージ | 2MB | **記載なし** | ✅/❌ |
| トレーニング時間 | **記載なし** | 約1時間 | ❌/✅ |
| VRAM使用量 | **記載なし** | **記載なし** | ❌❌ |

---

## 4. 確実に言えること（論文に基づく）

### 4.1 Deformable 3D Gaussians

1. ✅ **RTX 3090で実験**
2. ✅ **250k以下のガウシアンで30+ FPS**
3. ✅ **変形ネットワークは2MBの追加ストレージ**
4. ✅ **8層、256次元のMLPネットワーク**
5. ✅ **800×800解像度（合成データ）**
6. ✅ **ガウシアン数が多いとメモリ消費増加**
7. ❌ **VRAM使用量の具体的な数値は記載なし**

### 4.2 OmniRe

1. ✅ **RTX 4090で実験、約1時間/シーン**
2. ✅ **背景初期化で1,000,000ポイント**
3. ✅ **640×960解像度、3カメラ**
4. ✅ **SH次数: 背景/車両/変形可能ノードで3、SMPLノードで1**
5. ✅ **Absolute Gradient Gaussiansでメモリ制御**
6. ✅ **DeformableGSはAbsGradなしでOOM**
7. ❌ **VRAM使用量の具体的な数値は記載なし**

---

## 5. 実用的な推奨事項（論文根拠あり）

### 5.1 ガウシアン数の管理

**Deformable 3D Gaussians**の記載より:
> "when the number of 3D Gaussians is below 250k, our method can achieve real-time rendering over 30 FPS"

**推奨**:
- ✅ リアルタイムレンダリングを目指す場合: **250k未満**を維持
- ✅ これを超える場合: パフォーマンス低下とメモリ増加を覚悟

---

### 5.2 メモリ制御戦略（OmniRe）

**論文記載**:
> "we utilize the absolute gradient Gaussians as introduced in Ye et al. (2024) to control memory usage"

> "DeformableGS fails to run due to out-of-memory issues when AbsGrad is disabled"

**推奨**:
- ✅ **Absolute Gradient Gaussiansの使用は必須**（特に複雑なシーン）
- ✅ これなしではOOMのリスクが高い

---

### 5.3 球面調和関数の次数調整（OmniRe）

**論文記載**:
> "The degrees of spherical harmonics are set to 3 for background, rigid nodes, and non-rigid deformable nodes, while it is set to 1 for non-rigid SMPL nodes."

**推奨**:
- ✅ 重要なオブジェクト（背景、車両）: **次数3**
- ✅ 細かいオブジェクト（歩行者）: **次数1で効率化**
- ✅ メモリ削減効果: 192B → 48B（75%削減）

---

### 5.4 ネットワークサイズの影響（Deformable）

**論文記載**:
> "our additional storage compared to 3D Gaussians is only 2MB"

**推奨**:
- ✅ 変形ネットワークのメモリは**ガウシアン本体に比べて無視できる**
- ✅ ネットワークサイズの調整よりもガウシアン数の管理が重要

---

## 6. 結論

### ✅ 論文で確認できる事実

1. **Deformable 3D Gaussians**:
   - RTX 3090使用
   - 250k以下で30+ FPS
   - 変形ネットワーク: 2MB
   - ガウシアン数増加でメモリ消費増加

2. **OmniRe**:
   - RTX 4090使用、1時間/シーン
   - 背景1,000,000ポイント
   - SH次数: 3 or 1（ノードにより）
   - Absolute Gradientでメモリ制御
   - AbsGradなしでOOMリスク

### ⚠️ 推測・計算に基づく情報

1. **ガウシアンあたりのメモリ**:
   - SH次数3: 約236 bytes（推測）
   - SH次数1: 約92 bytes（推測）

2. **トレーニング時の総VRAM**:
   - Deformable: 1-2 GB（推測）
   - OmniRe: 3-5 GB（推測、複雑なシーン）

3. **倍率**:
   - オプティマイザとバッファで5-10倍（推測）

### 📌 最も重要な教訓

**論文から明確に言えること**:
1. **ガウシアン数が最大のボトルネック**（両論文で言及）
2. **Absolute Gradient Gaussiansは必須**（OmniRe、特に複雑なシーン）
3. **SH次数の戦略的調整が有効**（OmniRe）
4. **変形ネットワークのメモリは小さい**（Deformable）

**具体的なVRAM使用量については論文に記載がない**ため、実装時は実測が必要です。
