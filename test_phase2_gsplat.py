#!/usr/bin/env python3
"""
Phase 2 検証: gsplat 1.4.0 の互換性確認
特に cuda_legacy API が利用可能かを確認
"""

import sys

def test_gsplat():
    """gsplatの基本動作とcuda_legacy API確認"""
    print("=" * 60)
    print("Phase 2: gsplat 1.4.0 互換性確認")
    print("=" * 60)

    try:
        import gsplat
        print(f"\n✅ gsplat version: {gsplat.__version__}")

        # 期待バージョン確認
        expected_version = "1.4.0"
        version_ok = gsplat.__version__.startswith(expected_version)
        print(f"{'✅' if version_ok else '❌'} バージョン確認: {gsplat.__version__} (期待: {expected_version})")

        print("\n" + "-" * 60)
        print("drivestudioで使用しているAPI確認:")
        print("-" * 60)

        # drivestudioで使用されているgsplat APIの確認
        apis_status = {}

        # 1. cuda_legacy API (重要！)
        try:
            from gsplat.cuda_legacy._wrapper import num_sh_bases
            print("✅ gsplat.cuda_legacy._wrapper.num_sh_bases")
            apis_status['num_sh_bases'] = True
        except ImportError as e:
            print(f"❌ gsplat.cuda_legacy._wrapper.num_sh_bases: {e}")
            apis_status['num_sh_bases'] = False

        try:
            from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
            print("✅ gsplat.cuda_legacy._torch_impl.quat_to_rotmat")
            apis_status['quat_to_rotmat'] = True
        except ImportError as e:
            print(f"❌ gsplat.cuda_legacy._torch_impl.quat_to_rotmat: {e}")
            apis_status['quat_to_rotmat'] = False

        # 2. 通常のAPI
        try:
            from gsplat.rendering import rasterization
            print("✅ gsplat.rendering.rasterization")
            apis_status['rasterization'] = True
        except ImportError as e:
            print(f"❌ gsplat.rendering.rasterization: {e}")
            apis_status['rasterization'] = False

        try:
            from gsplat.cuda._wrapper import spherical_harmonics
            print("✅ gsplat.cuda._wrapper.spherical_harmonics")
            apis_status['spherical_harmonics'] = True
        except ImportError as e:
            print(f"❌ gsplat.cuda._wrapper.spherical_harmonics: {e}")
            apis_status['spherical_harmonics'] = False

        print("\n" + "-" * 60)
        print("代替API確認（cuda_legacyが使えない場合）:")
        print("-" * 60)

        # 代替API候補
        alternative_apis = {}

        try:
            from gsplat.utils import normalized_quat_to_rotmat
            print("✅ gsplat.utils.normalized_quat_to_rotmat (quat_to_rotmatの代替候補)")
            alternative_apis['normalized_quat_to_rotmat'] = True
        except (ImportError, AttributeError) as e:
            print(f"❌ gsplat.utils.normalized_quat_to_rotmat: {e}")
            alternative_apis['normalized_quat_to_rotmat'] = False

        print("\n" + "=" * 60)
        print("判定結果:")
        print("=" * 60)

        # cuda_legacy APIが使えるか
        cuda_legacy_ok = apis_status['num_sh_bases'] and apis_status['quat_to_rotmat']

        if cuda_legacy_ok:
            print("✅ cuda_legacy API: 利用可能")
            print("   → drivestudioのコード修正は不要です")
            return True
        else:
            print("❌ cuda_legacy API: 利用不可")
            print("   → 代替実装が必要です")

            # 代替案の提示
            print("\n" + "-" * 60)
            print("代替実装の方針:")
            print("-" * 60)

            if not apis_status['num_sh_bases']:
                print("• num_sh_bases の代替:")
                print("  def num_sh_bases(degree: int) -> int:")
                print("      return (degree + 1) ** 2")

            if not apis_status['quat_to_rotmat']:
                print("• quat_to_rotmat の代替:")
                if alternative_apis.get('normalized_quat_to_rotmat'):
                    print("  from gsplat.utils import normalized_quat_to_rotmat")
                else:
                    print("  from pytorch3d.transforms import quaternion_to_matrix")

            return False

    except ImportError as e:
        print(f"\n❌ gsplat Import Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gsplat()
    print("\n" + "=" * 60)
    if success:
        print("🎉 Phase 2: cuda_legacy API が利用可能です！")
    else:
        print("⚠️  Phase 2: 代替実装が必要です")
    print("=" * 60)
    sys.exit(0 if success else 1)
