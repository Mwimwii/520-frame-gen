# Intel HD Graphics 520 Frame Generation Research

## Overview

This repository contains research and proof-of-concept tools for enabling **frame generation (frame interpolation) on Intel HD Graphics 520** - an integrated GPU from 2015 (Gen9/Skylake architecture) that does **not** support any commercial frame generation technology (DLSS 3, FSR 3, XeSS 3 MFG).

The **core hypothesis** is that the HD 520's overlooked **VME (Video Motion Estimation) engine** - part of Intel Quick Sync Video - can be repurposed for hardware-accelerated motion estimation, which is the fundamental bottleneck in frame interpolation algorithms.

## Key Findings

### Hardware Specifications
- **Architecture**: Gen9 (Skylake), 24 Execution Units (EUs)
- **Performance**: 403.2 GFLOPS (FP32 peak)
- **Release**: 2015

### Commercial Support Status
| Technology | Supports HD 520? | Reason |
|---|---|---|
| NVIDIA DLSS 3 Frame Gen | **No** | Requires RTX 40xx Tensor cores + Optical Flow |
| AMD FSR 3 Frame Gen | **No** | Requires Intel Arc / RDNA2+ / RTX 20+ |
| Intel XeSS 3 MFG | **No** | Exclusive to Intel Arc discrete GPUs |

### Overlooked Hardware Features (The Hypothesis)
1. **VME Engine**: Hardware motion estimation for video encoding - the core of frame interpolation
2. **VEBox (Video Enhancement Box)**: Video post-processing pipeline (scaling, sharpening, deinterlacing)
3. **OpenCL 3.0**: Compute engine (24 EUs) for motion-compensated frame blending
4. **SFC (Scalar & Format Converter)**: Hardware-accelerated frame output

## Files

### Research Documentation
- **`RESEARCH.md`** - Full research report with hardware specs, frame generation technology comparison, and the hypothesis about overlooked VME/VEBox capabilities

### Test Tools
- **`tests/gpu_capability_test.py`** - Detects GPU, OpenCL, VA-API (VME), Intel Media SDK, and FFmpeg capabilities
- **`tests/run_full_test.sh`** - Complete test suite covering all detection methods
- **`poc_frame_interpolation.py`** - Proof-of-concept motion-compensated frame interpolation using OpenCV optical flow (with optional OpenCL acceleration)

## Quick Start

### 1. Run the Full Test Suite
```bash
./tests/run_full_test.sh
```

### 2. Check GPU Capabilities
```bash
python3 tests/gpu_capability_test.py
```

### 3. Run the Frame Interpolation Demo
```bash
python3 poc_frame_interpolation.py --demo
```

### 4. Benchmark Performance
```bash
python3 poc_frame_interpolation.py --benchmark
```

### 5. Process a Video File
```bash
python3 poc_frame_interpolation.py input.mp4 -o output.mp4 --fps 60
```

## Requirements

### Ubuntu/Debian
```bash
sudo apt install clinfo vainfo ffmpeg intel-media-va-driver-non-free
pip install opencv-python numpy scipy
```

### Windows
- Install Intel Graphics Driver
- Install OpenCL runtime
- `pip install opencv-python numpy scipy`

## Technical Approach

The proof-of-concept implements a three-stage pipeline:

1. **Motion Estimation**: Calculate motion vectors between consecutive frames using optical flow (Farneback algorithm). In a hardware-accelerated version, this would use the VME engine via VA-API/Intel Media SDK.

2. **Motion Compensation**: Warp both frames using the motion vectors and blend them to create an intermediate frame. This uses OpenCL compute when available.

3. **Post-processing**: Optional sharpening and noise reduction using VEBox-like operations.

The VME engine in Quick Sync Video can accelerate step 1 by orders of magnitude, as it's dedicated hardware for exactly this computation.

## References

- [Intel Graphics Technology](https://en.wikipedia.org/wiki/Intel_Graphics_Technology) (Wikipedia)
- [Intel Arc](https://en.wikipedia.org/wiki/Intel_Arc) (Wikipedia)
- [GPUOpen - FSR 3](https://gpuopen.com/) (AMD)
- [Intel Compute Runtime](https://github.com/intel/compute-runtime) (OpenCL/Level Zero)
- [Intel Media Driver](https://github.com/intel/media-driver) (VA-API/VAAPI)
- [FFmpeg minterpolate](https://ffmpeg.org/ffmpeg-filters.html#minterpolate)
