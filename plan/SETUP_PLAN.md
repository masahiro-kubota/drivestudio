# SplatAD 環境構築計画

## 目標
splatadをnuscenes-gs-labと同じ依存関係（torch 2.1.2 + CUDA 11.8）で動作させる

## フェーズ1: 環境確認 🔍

### Step 1.1: CUDAバージョン確認
- [x] `nvcc --version`でCUDAコンパイラのバージョン確認
- [x] `nvidia-smi`でドライバーとランタイムバージョン確認
- [x] CUDA 11.8が利用可能か確認
- [x] 結果をこのファイルに記録

**結果：**
```
nvidia-smi出力:
✅ GPU: NVIDIA GeForce RTX 4070 Ti (12GB VRAM)
✅ Driver Version: 590.48.01
✅ CUDA Version (Driver): 13.1

nvcc確認:
❌ nvcc: Not found
❌ CUDA Toolkit: インストールされていない
❌ /usr/local/cuda*: 存在しない

現在のPyTorch状態:
- Version: 1.8.0a0 (古い開発版)
- CUDA available: False
- CUDA version: None
```

**問題点:**
⚠️ CUDA Toolkit（nvccコンパイラ）がインストールされていない
⚠️ gsplatをソースからビルドするにはCUDA Toolkit 11.8が必須
⚠️ 現在のPyTorchもCUDAサポートなし

**次のアクション:**
CUDA Toolkit 11.8のインストールが必要（またはCUDA 11.xならOK）

### Step 1.2: 現在のPython環境確認
- [x] `python --version`
- [x] `which python`
- [x] 仮想環境の状態確認

**結果：**
```
✅ Python: 3.10.12 (要件: >=3.10,<3.11 を満たす)
✅ Path: /usr/local/bin/python
✅ uv: 0.5.24 インストール済み
✅ Prefix: /usr (システムPython)

仮想環境: .venv なし（これから作成）
```

### Step 1.3: 既存のPyTorch確認
- [ ] 既にPyTorchがインストールされているか確認
- [ ] バージョンとCUDAサポート確認

**結果：**
```
（ここに結果を記録）
```

---

## フェーズ1.5: CUDA Toolkit 11.8のインストール 🔧

### Step 1.5.1: インストール方法の選択

**現在の状況:**
- Ubuntu 22.04.5 LTS
- NVIDIAドライバー 590.48.01 既にインストール済み
- CUDA Toolkitのみが必要

**推奨方法: runfileインストーラー**

```bash
# 1. インストーラーをダウンロード
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 2. 実行権限を付与
chmod +x cuda_11.8.0_520.61.05_linux.run

# 3. インストール（ドライバーは除外）
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override

# 注意: インストール時に「Driver」のチェックを外す
# --toolkit オプションでToolkitのみをインストール
```

### Step 1.5.2: 環境変数の設定

```bash
# ~/.bashrcまたは~/.zshrcに追加
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 1.5.3: インストール確認

```bash
nvcc --version
# 期待される出力: Cuda compilation tools, release 11.8
```

**インストール結果:**
```
✅ Driver:   Not Selected （正しい）
✅ Toolkit:  Installed in /usr/local/cuda-11.8/

⚠️ WARNING: Incomplete installation! This installation did not install the CUDA Driver.
   → これは予想通り。既存ドライバー 590.48.01 (>= 520.00) があるので問題なし
```

**環境変数設定:**
```bash
# ~/.zshrc に追加済み
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
```

**nvcc動作確認:**
```
✅ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Sep_21_10:33:58_PDT_2022
Cuda compilation tools, release 11.8, V11.8.89
Build cuda_11.8.r11.8/compiler.31833905_0
```

---

## フェーズ2: pyproject.toml調整 📝

### Step 2.1: CUDAバージョンとの整合性確認
- [ ] pyproject.tomlのCUDAバージョン（cu118）が実際のCUDAと合っているか確認
- [ ] 必要に応じて修正

### Step 2.2: 依存関係の検証
- [ ] torch 2.1.2がCUDA 11.8をサポートしているか確認
- [ ] PyTorchの公式wheelが利用可能か確認

---

## フェーズ3: 基本パッケージのインストール 📦

### Step 3.1: uv自体の確認
- [ ] `uv --version`でuvがインストールされているか確認
- [ ] 必要に応じてuvをインストール

### Step 3.2: uv syncの準備
- [ ] .venvディレクトリの確認
- [ ] 古い環境があれば削除するか判断

### Step 3.3: uv sync実行（CUDA拡張なし）
- [ ] まずはPython依存関係のみインストール
- [ ] `BUILD_NO_CUDA=1`または依存関係のみ
- [ ] エラーがないか確認

**実行コマンド：**
```bash
uv sync
```

---

## フェーズ4: 動作確認（CUDAなし） ✅

### Step 4.1: 基本的なimportテスト
- [ ] `python -c "import torch; print(torch.__version__)"`
- [ ] `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] `python -c "import torch; print(torch.version.cuda)"`

### Step 4.2: gsplatのPython部分のimportテスト
- [ ] `python -c "import gsplat"`（CUDA関数を使わない）
- [ ] エラーメッセージを確認

---

## フェーズ5: CUDA拡張のビルド 🔨

### Step 5.1: ビルド環境の確認
- [ ] `ninja --version`
- [ ] CUDAコンパイラへのパス確認
- [ ] 必要な開発ツールの確認

### Step 5.2: ビルドの実行
- [ ] `BUILD_CUDA=1 uv pip install -e .`を実行
- [ ] ビルドログを保存
- [ ] エラーがあれば原因特定

### Step 5.3: ビルド結果の確認
- [ ] `.so`ファイルが生成されたか確認
- [ ] `gsplat/cuda/csrc/`内のファイル確認

---

## フェーズ6: 最終動作確認 🎯

### Step 6.1: CUDA機能のテスト
- [ ] `python -c "from gsplat.rendering import rasterization; print('OK')"`
- [ ] `python -c "from gsplat.rendering import lidar_rasterization; print('OK')"`

### Step 6.2: サンプルコードの実行
- [ ] examples/lidar_rendering.ipynbの最初のセルを実行
- [ ] エラーがないか確認

### Step 6.3: テストスイートの実行
- [ ] `pytest tests/test_rasterization.py -v`
- [ ] GPU必要なテストが通るか確認

---

## トラブルシューティング記録

### 問題1:
**現象：**
```
（問題が発生したらここに記録）
```

**解決方法：**
```
（解決策を記録）
```

---

## 成功時の最終環境

```
Python:
PyTorch:
CUDA:
gsplat:
```

## 次のステップ

- [ ] nuScenesデータのセットアップ
- [ ] neurad-studioとの統合
- [ ] 実際のデータで動作確認
