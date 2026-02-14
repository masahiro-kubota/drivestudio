# 再構築計画

## 目的
pkg_resources問題を解決してCUDA拡張を動かす

## Phase 0: 根本原因特定（5分）

### Test 1: uvでpkg_resourcesは使えるのか？
```bash
# 新しいテスト環境で確認
cd /tmp
mkdir test-uv-pkg && cd test-uv-pkg
uv init --no-workspace
uv add setuptools
uv run python -c "import pkg_resources; print('OK')"
```

**期待結果:**
- ✅ OK → uvでも使える（設定の問題）
- ❌ エラー → uvの構造的問題

### Test 2: 通常のvenv+pipなら？
```bash
cd /tmp/test-uv-pkg
python -m venv test-venv
source test-venv/bin/activate
pip install setuptools
python -c "import pkg_resources; print('OK')"
deactivate
```

**期待結果:**
- ✅ OK → venv+pipなら動く
- ❌ エラー → システム環境の問題

---

## Phase 1: 解決策の選択（診断結果による）

### ケースA: uvでも動く場合
→ 現在の.venvに問題がある
→ `.venv`削除 + `uv sync`やり直し

### ケースB: venv+pipなら動く場合
→ uvの問題
→ venv+pipに切り替える

### ケースC: どちらもダメ
→ システム環境の問題
→ 別の解決策を探す

---

## Phase 2: 実行前の最終確認

**実行する前に:**
1. [ ] 診断結果を確認
2. [ ] 選択した解決策のリスクを理解
3. [ ] バックアップ不要か確認（ソースコードは変更していない）
4. [ ] 所要時間の見積もり

**実行時の fail fast ポイント:**
- setuptools installで5秒以内に完了するか
- pkg_resources importで即座に成功するか
- uv sync/pip installで3分以内に完了するか
- 各ステップでエラーが出たら即座に停止

---

## Phase 3: 実行（診断結果により決定）

未定（Phase 0の結果を見てから）

---

## 成功の定義

```python
# これが動けば成功
import torch
from gsplat.rendering import rasterization

means = torch.rand(10, 3, device='cuda')
quats = torch.rand(10, 4, device='cuda')
scales = torch.rand(10, 3, device='cuda')
opacities = torch.rand(10, device='cuda')
colors = torch.rand(10, 3, device='cuda')
viewmats = torch.eye(4, device='cuda').unsqueeze(0)
Ks = torch.tensor([[300.0, 0, 150], [0, 300, 100], [0, 0, 1]], device='cuda').unsqueeze(0)

out, _, _ = rasterization(means, quats, scales, opacities, colors, None, viewmats, Ks, 300, 200)
print(f'Success: {out.shape}')
```

**タイムリミット: 各Phase最大10分**
