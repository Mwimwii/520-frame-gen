#!/usr/bin/env python3
"""
Intel HD Graphics 520 - GPU Capability Detection Script

This script detects and reports the GPU hardware capabilities relevant to
frame generation support on Intel HD Graphics 520.

Run: python3 tests/gpu_capability_test.py
"""

import subprocess
import sys
import platform
import json
import re


def run_command(cmd, timeout=10):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, shell=True
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", "Command not found"
    except Exception as e:
        return -1, "", str(e)


def detect_intel_gpu():
    """Detect Intel HD Graphics 520 (or other Intel GPUs) using lspci."""
    print("\n[INFO] Detecting Intel GPU via lspci...")
    rc, stdout, stderr = run_command("lspci -v")
    if rc != 0:
        print(f"[WARN] lspci not available or failed: {stderr}")
        return None

    gpus = []
    current_gpu = None

    for line in stdout.split("\n"):
        if "VGA compatible controller" in line or "Display controller" in line:
            match = re.search(r"Intel.*Graphics.*", line, re.IGNORECASE)
            if match:
                current_gpu = match.group(0)
                gpus.append({"vendor": "Intel", "model": current_gpu, "is_hd520": "520" in current_gpu or "skylake" in current_gpu.lower()})
    return gpus


def detect_opencl():
    """Detect OpenCL availability and version using clinfo."""
    print("\n[INFO] Checking OpenCL availability via clinfo...")
    rc, stdout, stderr = run_command("clinfo --help", timeout=3)

    if rc != 0:
        print("[WARN] clinfo not found. Install with: apt install clinfo")
        return {"available": False, "error": "clinfo not installed"}

    rc, stdout, stderr = run_command("clinfo", timeout=30)
    if rc != 0:
        return {"available": False, "error": stderr or stdout}

    platforms = []
    lines = stdout.split("\n")
    current_platform = None

    for line in lines:
        if "Number of platforms" in line:
            count = int(re.search(r"(\d+)", line).group(1))
            print(f"  Platforms found: {count}")
        if "Platform Name" in line:
            name = line.split(":", 1)[1].strip()
            current_platform = {"name": name, "devices": []}
            platforms.append(current_platform)
        if "Device Name" in line and current_platform:
            device_name = line.split(":", 1)[1].strip()
            current_platform["devices"].append({"name": device_name})
        if "CL_DEVICE_VERSION" in line or "OpenCL" in line:
            ver_match = re.search(r"OpenCL\s+([\d.]+)", line)
            if ver_match and current_platform and current_platform["devices"]:
                current_platform["devices"][-1]["opencl_version"] = ver_match.group(1)

    return {"available": len(platforms) > 0, "platforms": platforms, "raw_output": stdout[:2000]}


def detect_vulkan():
    """Detect Vulkan support using vulkaninfo."""
    print("\n[INFO] Checking Vulkan availability...")
    rc, stdout, stderr = run_command("vulkaninfo --summary", timeout=10)

    if rc != 0:
        print("[INFO] Vulkan not available (may not be installed)")
        return {"available": False}

    gpu_count = stdout.count("GPU id")
    print(f"  Vulkan GPUs detected: {gpu_count}")

    return {"available": gpu_count > 0, "gpu_count": gpu_count, "raw": stdout[:1500]}


def detect_vaapi():
    """Detect VA-API / Intel Media SDK availability."""
    print("\n[INFO] Checking VA-API availability...")
    rc, stdout, stderr = run_command("vainfo", timeout=10)

    if rc != 0:
        print("[INFO] VA-API (vainfo) not available")
        return {"available": False, "error": stderr or "vainfo not installed"}

    profiles = re.findall(r"VAProfile(\w+)", stdout)
    entrypoints = re.findall(r"VAEntrypoint(\w+)", stdout)

    vme_capable = any("EncSlice" in ep or "EncPicture" in ep for ep in entrypoints)

    return {
        "available": True,
        "profiles": list(set(profiles)),
        "entrypoints": list(set(entrypoints)),
        "vme_capable": vme_capable,
        "raw": stdout[:1500],
    }


def detect_intel_media_sdk():
    """Check for Intel Media SDK / oneVPL library presence."""
    print("\n[INFO] Checking Intel Media SDK / oneVPL...")

    libs = ["libmfx.so", "libmfxhw64.so", "libvpl.so", "libmfx-gen.so"]
    found = []

    for lib in libs:
        rc, stdout, _ = run_command(f"ldconfig -p | grep {lib}")
        if stdout:
            found.append(lib)
            print(f"  Found: {lib}")

    return {
        "available": len(found) > 0,
        "libraries_found": found,
    }


def detect_ffmpeg_capabilities():
    """Check FFmpeg capabilities for frame interpolation filters."""
    print("\n[INFO] Checking FFmpeg frame interpolation support...")
    rc, stdout, _ = run_command("ffmpeg -hide_banner -filters")

    if rc != 0:
        print("[WARN] FFmpeg not found")
        return {"available": False}

    filters_of_interest = ["minterpolate", "interp", "fps", "setpts", "tblend"]
    available_filters = []
    for f in filters_of_interest:
        if f" {f} " in stdout or f"\n{f} " in stdout:
            available_filters.append(f)
            print(f"  FFmpeg filter available: {f}")

    rc2, opencl_info, _ = run_command("ffmpeg -hide_banner -hwaccels")
    hwaccels = []
    if opencl_info:
        for line in opencl_info.split("\n"):
            if "opencl" in line.lower():
                hwaccels.append(line.strip())
                print(f"  HWAccel: {line.strip()}")

    return {
        "available": len(available_filters) > 0,
        "interp_filters": available_filters,
        "hwaccels": hwaccels,
    }


def detect_directx():
    """Check DirectX runtime (Windows only)."""
    if platform.system() != "Windows":
        print("\n[INFO] Skipping DirectX detection (non-Windows)")
        return {"available": False, "note": "Windows only"}

    print("\n[INFO] Checking DirectX support...")
    rc, stdout, stderr = run_command("dxdiag /t dxdiag_output.txt > nul 2>&1 && type dxdiag_output.txt")
    return {"available": rc == 0, "raw": stdout[:1500] if stdout else ""}


def assess_frame_generation_support():
    """Assess overall frame generation support level."""
    print("\n" + "=" * 60)
    print("FRAME GENERATION SUPPORT ASSESSMENT")
    print("=" * 60)

    intel_gpu = detect_intel_gpu()
    opencl_info = detect_opencl()
    vulkan_info = detect_vulkan()
    vaapi_info = detect_vaapi()
    media_sdk_info = detect_intel_media_sdk()
    ffmpeg_info = detect_ffmpeg_capabilities()

    print("\n--- Summary ---")
    print(f"Intel GPU Detected: {intel_gpu is not None and len(intel_gpu) > 0}")
    if intel_gpu:
        for gpu in intel_gpu:
            print(f"  - {gpu['model']} (HD 520 match: {gpu['is_hd520']})")

    print(f"OpenCL Available: {opencl_info.get('available', False)}")
    print(f"Vulkan Available: {vulkan_info.get('available', False)}")
    print(f"VA-API Available: {vaapi_info.get('available', False)}")
    print(f"  VME Engine Accessible: {vaapi_info.get('vme_capable', False)}")
    print(f"Intel Media SDK Available: {media_sdk_info.get('available', False)}")
    print(f"FFmpeg Interpolation Filters: {ffmpeg_info.get('available', False)}")

    results = {
        "intel_gpu": intel_gpu,
        "opencl": opencl_info,
        "vulkan": vulkan_info,
        "vaapi": vaapi_info,
        "media_sdk": media_sdk_info,
        "ffmpeg": ffmpeg_info,
    }

    print("\n--- Frame Generation Feasibility ---")
    can_do_vme = vaapi_info.get("available", False) and vaapi_info.get("vme_capable", False)
    can_do_opencl = opencl_info.get("available", False)
    can_do_ffmpeg = ffmpeg_info.get("available", False)

    if can_do_vme and can_do_opencl:
        print("  [EXCELLENT] VME + OpenCL path available!")
        print("  -> Can use hardware motion estimation (VME)")
        print("  -> Can use OpenCL for motion-compensated interpolation")
        print("  -> Can use FFmpeg minterpolate as fallback")
    elif can_do_ffmpeg:
        print("  [GOOD] FFmpeg interpolation path available")
        print("  -> Can use FFmpeg minterpolate for frame generation")
        print("  -> May be slower without hardware VME acceleration")
    elif can_do_opencl:
        print("  [MODERATE] OpenCL path available")
        print("  -> Can implement software-based motion estimation")
        print("  -> No hardware VME acceleration")
    else:
        print("  [LIMITED] No GPU acceleration path detected")
        print("  -> CPU-only frame interpolation will be very slow")

    report_path = "gpu_capability_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Output] Full report saved to {report_path}")

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("Intel HD Graphics 520 - GPU Capability Detection")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version.split()[0]}")
    print("=" * 60)

    assess_frame_generation_support()
