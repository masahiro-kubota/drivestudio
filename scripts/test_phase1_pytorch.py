#!/usr/bin/env python3
"""
Phase 1 検証: PyTorch 2.1.2 + CUDA 11.8 の動作確認
"""

import sys

def test_pytorch():
    """PyTorchの基本動作を確認"""
    print("=" * 60)
    print("Phase 1: PyTorch環境の検証")
    print("=" * 60)

    try:
        import torch
        import torchvision

        print(f"\n✅ PyTorch version: {torch.__version__}")
        print(f"✅ torchvision version: {torchvision.__version__}")

        # CUDA確認
        cuda_available = torch.cuda.is_available()
        print(f"\n{'✅' if cuda_available else '❌'} CUDA available: {cuda_available}")

        if cuda_available:
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ Current device: {torch.cuda.current_device()}")
            print(f"✅ Device name: {torch.cuda.get_device_name(0)}")

            # 簡単なテンソル演算
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x @ y
            print(f"\n✅ GPU tensor operation successful: {z.shape}")

        # バージョンチェック
        expected_torch_version = "2.1.2"
        expected_torchvision_version = "0.16.2"

        torch_ok = torch.__version__.startswith(expected_torch_version)
        torchvision_ok = torchvision.__version__.startswith(expected_torchvision_version)

        print("\n" + "=" * 60)
        print("バージョン確認:")
        print("=" * 60)
        print(f"{'✅' if torch_ok else '❌'} torch: {torch.__version__} (期待: {expected_torch_version})")
        print(f"{'✅' if torchvision_ok else '❌'} torchvision: {torchvision.__version__} (期待: {expected_torchvision_version})")
        print(f"{'✅' if cuda_available else '❌'} CUDA: {cuda_available}")

        if torch_ok and torchvision_ok and cuda_available:
            print("\n" + "=" * 60)
            print("🎉 Phase 1: すべてのチェックが成功しました！")
            print("=" * 60)
            return True
        else:
            print("\n" + "=" * 60)
            print("⚠️  Phase 1: いくつかのチェックが失敗しました")
            print("=" * 60)
            return False

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        return False

if __name__ == "__main__":
    success = test_pytorch()
    sys.exit(0 if success else 1)
