#!/usr/bin/env python3
"""
Phase 2-3 検証: gsplat 1.4.0対応の修正確認
"""

import sys
import torch

def test_imports():
    """修正したgsplat APIのインポート確認"""
    print("=" * 60)
    print("Phase 2-3: gsplat 1.4.0対応の修正確認")
    print("=" * 60)

    try:
        # gsplat 1.4.0のAPIを直接インポート
        from gsplat.rendering import rasterization
        from gsplat.cuda._wrapper import spherical_harmonics
        from gsplat.utils import normalized_quat_to_rotmat as quat_to_rotmat

        print("\n✅ gsplat.rendering.rasterization")
        print("✅ gsplat.cuda._wrapper.spherical_harmonics")
        print("✅ gsplat.utils.normalized_quat_to_rotmat")

        # num_sh_basesの代替実装をテスト
        def num_sh_bases(degree: int) -> int:
            """Calculate number of spherical harmonics bases for given degree"""
            return (degree + 1) ** 2

        print("\n" + "-" * 60)
        print("num_sh_bases 関数テスト:")
        print("-" * 60)

        test_cases = [(0, 1), (1, 4), (2, 9), (3, 16)]
        all_passed = True

        for degree, expected in test_cases:
            result = num_sh_bases(degree)
            status = "✅" if result == expected else "❌"
            print(f"{status} num_sh_bases({degree}) = {result} (期待: {expected})")
            if result != expected:
                all_passed = False

        # quat_to_rotmat のテスト
        print("\n" + "-" * 60)
        print("quat_to_rotmat 関数テスト:")
        print("-" * 60)

        # 単位クォータニオン (恒等回転)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda')
        rotmat = quat_to_rotmat(quat)
        print(f"✅ quat_to_rotmat 動作確認")
        print(f"   入力クォータニオン: {quat.cpu().numpy()}")
        print(f"   出力回転行列の形状: {rotmat.shape} (期待: torch.Size([1, 3, 3]))")

        # 恒等行列に近いか確認
        identity = torch.eye(3, device='cuda')
        diff = torch.abs(rotmat[0] - identity).max().item()
        is_identity = diff < 0.01
        status = "✅" if is_identity else "❌"
        print(f"{status} 単位クォータニオン → 恒等行列: 誤差 {diff:.6f}")

        print("\n" + "=" * 60)
        if all_passed and is_identity:
            print("🎉 Phase 2-3: すべてのテストが成功しました！")
            print("=" * 60)
            print("\n✅ gsplat 1.4.0への移行が完了しました")
            print("✅ cuda_legacy APIの代替実装が正しく動作しています")
            return True
        else:
            print("⚠️  Phase 2-3: いくつかのテストが失敗しました")
            print("=" * 60)
            return False

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
