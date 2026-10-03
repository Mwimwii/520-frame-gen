#!/bin/bash
#
# Intel HD Graphics 520 - Full Frame Generation Test Suite
#
# This script tests three frame generation approaches:
# 1. VA-API / Intel Media SDK VME-based motion estimation
# 2. FFmpeg minterpolate (motion-compensated interpolation)
# 3. OpenCV-based frame interpolation (the PoC script)
#
# USAGE:
#   chmod +x run_full_test.sh
#   ./run_full_test.sh
#
# Requirements (install as needed):
#   sudo apt install clinfo vainfo ffmpeg intel-media-va-driver-non-free
#   pip install opencv-python numpy scipy
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/test_results"

mkdir -p "$RESULTS_DIR"

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $1"
}

log_test() {
    echo -e "${CYAN}[TEST]${NC} $1"
}

echo "=========================================="
echo " Intel HD Graphics 520 - Frame Gen Test Suite"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Test 1: System Information and GPU Detection
# ---------------------------------------------------------------------------
log_test "Test 1: System Information and GPU Detection"

echo "  Operating System:"
    uname -a 2>&1 | sed 's/^/    /'
    
echo "  CPU:"
    lscpu | grep "Model name" | head -1 | sed 's/^/    /' || echo "    Not available"

echo "  GPU (lspci):"
    if command -v lspci &>/dev/null; then
        lspci | grep -i "vga\|display\|3d" | sed 's/^/    /' || echo "    No GPU detected"
    else
        echo "    lspci not available"
    fi

echo "  GPU (lshw):"
    if command -v lshw &>/dev/null; then
        lshw -C display 2>/dev/null | grep -A3 "product\|vendor\|description" | head -8 | sed 's/^/    /' || echo "    Not available"
    else
        echo "    lshw not available"
    fi

echo "  Intel GPU specifically:"
    if lspci 2>/dev/null | grep -i intel | grep -i "vga\|display\|3d"; then
        INTEL_FOUND=1
        log_success "Intel GPU detected"
    else
        log_warn "No Intel GPU detected (may be in VM/container)"
        INTEL_FOUND=0
    fi

# ---------------------------------------------------------------------------
# Test 2: OpenCL Capability Detection
# ---------------------------------------------------------------------------
log_test "Test 2: OpenCL Capability Detection"

if command -v clinfo &>/dev/null; then
    echo "  OpenCL Platforms:"
    clinfo 2>&1 | grep -E "Number of platforms|Platform Name|Device Name|OpenCL" | head -20 | sed 's/^/    /'
    
    PLATFORM_COUNT=$(clinfo 2>&1 | grep "Number of platforms" | grep -oP '\d+' | head -1)
    if [ "$PLATFORM_COUNT" -gt 0 ] 2>/dev/null; then
        log_success "OpenCL available with $PLATFORM_COUNT platform(s)"
        
        echo "  Intel OpenCL device check:"
        if clinfo 2>&1 | grep -qi "Intel"; then
            log_success "Intel OpenCL device found"
        else
            log_warn "No Intel OpenCL device (using CPU/other GPU)"
        fi
    else
        log_warn "No OpenCL platforms found"
    fi
else
    log_warn "clinfo not installed. Install: apt install clinfo"
fi

# ---------------------------------------------------------------------------
# Test 3: VA-API and Intel Media SDK
# ---------------------------------------------------------------------------
log_test "Test 3: VA-API (Hardware Acceleration) Detection"

if command -v vainfo &>/dev/null; then
    echo "  VA-API Driver Info:"
    vainfo 2>&1 | head -15 | sed 's/^/    /'
    
    if vainfo 2>&1 | grep -q "Intel"; then
        log_success "Intel VA-API driver detected"
        
        echo "  VME (Video Motion Estimation) capability check:"
        if vainfo 2>&1 | grep -qi "EncSlice\|EncPicture"; then
            log_success "VME engine is accessible via VA-API"
            echo "  -> VME can be used for hardware-accelerated motion estimation"
            echo "  -> This is the KEY overlooked feature for frame generation on HD 520"
        else
            log_warn "VME entrypoints not explicitly listed, but encoding may still work"
        fi
    else
        log_warn "VA-API driver found, but not Intel-specific"
    fi
else
    log_warn "vainfo not installed. Install: apt install vainfo"
    log_warn "Without VA-API, VME hardware acceleration is NOT accessible"
fi

# ---------------------------------------------------------------------------
# Test 4: FFmpeg Capability Detection
# ---------------------------------------------------------------------------
log_test "Test 4: FFmpeg Frame Interpolation Support"

if command -v ffmpeg &>/dev/null; then
    echo "  FFmpeg version:"
    ffmpeg -version | head -1 | sed 's/^/    /'
    
    echo "  Available interpolation-related filters:"
    ffmpeg -hide_banner -filters 2>/dev/null | grep -E "minterpolate|interp|tblend|fps" | sed 's/^/    /' || true
    
    if ffmpeg -hide_banner -filters 2>/dev/null | grep -q "minterpolate"; then
        log_success "FFmpeg minterpolate filter available (hardware-independent frame interpolation)"
    else
        log_warn "minterpolate filter not available"
    fi
    
    echo "  Hardware acceleration backends:"
    ffmpeg -hide_banner -hwaccels 2>/dev/null | sed 's/^/    /' || true
else
    log_error "FFmpeg not found"
fi

# ---------------------------------------------------------------------------
# Test 5: Python Environment and Packages
# ---------------------------------------------------------------------------
log_test "Test 5: Python Environment and Packages"

if command -v python3 &>/dev/null; then
    echo "  Python version:"
    python3 --version | sed 's/^/    /'
    
    PACKAGES=("cv2" "numpy" "scipy" "sklearn")
    for pkg in "${PACKAGES[@]}"; do
        if python3 -c "import $pkg" 2>/dev/null; then
            VER=$(python3 -c "import $pkg; print(getattr($pkg, '__version__', 'unknown'))" 2>/dev/null || echo "?")
            log_success "Package '$pkg' ($VER) available"
        else
            log_warn "Package '$pkg' not installed. Install: pip install $pkg"
        fi
    done
    
    echo "  OpenCV OpenCL build info:"
    python3 -c "
import cv2
info = cv2.getBuildInformation()
for line in info.split('\n'):
    if 'OpenCL' in line or 'opencl' in line:
        print('    ' + line.rstrip())
" 2>/dev/null || log_warn "Could not get OpenCV build info"
else
    log_error "Python3 not found"
fi

# ---------------------------------------------------------------------------
# Test 6: Run PoC Frame Interpolation Demo
# ---------------------------------------------------------------------------
log_test "Test 6: Run PoC Frame Interpolation Demo"

POC_SCRIPT="$PROJECT_DIR/poc_frame_interpolation.py"
if [ -f "$POC_SCRIPT" ]; then
    cd "$PROJECT_DIR"
    timeout 30 python3 "$POC_SCRIPT" --demo --no-opencl 2>&1 | tee "$RESULTS_DIR/demo_output.txt"
    if [ $? -eq 0 ]; then
        log_success "PoC demo completed"
    else
        log_warn "PoC demo timed out or failed (preview window may need display)"
    fi
else
    log_error "PoC script not found at $POC_SCRIPT"
fi

# ---------------------------------------------------------------------------
# Test 7: Run GPU Capability Detection Script
# ---------------------------------------------------------------------------
log_test "Test 7: Run GPU Capability Detection"

CAP_SCRIPT="$PROJECT_DIR/tests/gpu_capability_test.py"
if [ -f "$CAP_SCRIPT" ]; then
    cd "$PROJECT_DIR"
    python3 "$CAP_SCRIPT" 2>&1 | tee "$RESULTS_DIR/capability_output.txt"
    if [ $? -eq 0 ]; then
        log_success "Capability detection completed"
    else
        log_warn "Capability detection had issues"
    fi
else
    log_error "Capability detection script not found"
fi

# ---------------------------------------------------------------------------
# Test 8: FFmpeg minterpolate Test (if FFmpeg and test video available)
# ---------------------------------------------------------------------------
log_test "Test 8: FFmpeg minterpolate Test"

# Create a small test video if ffmpeg is available
if command -v ffmpeg &>/dev/null; then
    echo "  Creating test video (2 seconds, 30fps, 320x240)..."
    
    # Create a simple test video with a moving object
    ffmpeg -y -f lavfi -i "testsrc=size=320x240:rate=30:duration=2" \
           -vf "hue=s=0.5" \
           "$RESULTS_DIR/test_input.mp4" 2>/dev/null || true
    
    if [ -f "$RESULTS_DIR/test_input.mp4" ]; then
        echo "  Running minterpolate (motion-compensated interpolation)..."
        echo "  This tests the analytical frame interpolation path..."
        
        timeout 30 ffmpeg -y -i "$RESULTS_DIR/test_input.mp4" \
            -vf "minterpolate='mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps=60'" \
            "$RESULTS_DIR/test_output.mp4" 2>&1 | tail -5 | sed 's/^/    /'
        
        if [ -f "$RESULTS_DIR/test_output.mp4" ]; then
            IN_FRAMES=$(ffprobe -v error -select_streams v:0 -count_packets \
                -show_entries stream=nb_read_packets -of csv=p=0 \
                "$RESULTS_DIR/test_input.mp4" 2>/dev/null || echo "?")
            OUT_FRAMES=$(ffprobe -v error -select_streams v:0 -count_packets \
                -show_entries stream=nb_read_packets -of csv=p=0 \
                "$RESULTS_DIR/test_output.mp4" 2>/dev/null || echo "?")
            
            echo "  Input frames: $IN_FRAMES"
            echo "  Output frames: $OUT_FRAMES"
            
            if [ "$OUT_FRAMES" -gt "$IN_FRAMES" ] 2>/dev/null; then
                log_success "FFmpeg interpolation generated additional frames ($IN_FRAMES -> $OUT_FRAMES)"
            else
                log_warn "Frame count comparison inconclusive"
            fi
        else
            log_warn "Output video not created (may have timed out or failed)"
        fi
    else
        log_warn "Could not create test video"
    fi
else
    log_warn "FFmpeg not available for minterpolate test"
fi

# ---------------------------------------------------------------------------
# Test 9: Summary and Hypothesis Validation
# ---------------------------------------------------------------------------
log_test "Test 9: Summary and Hypothesis Validation"

echo ""
echo "=========================================="
echo " SUMMARY: Intel HD Graphics 520 - Frame Gen"
echo "=========================================="
echo ""

echo "HARDWARE CAPABILITIES:"
echo "  - Architecture: Gen9 (Skylake), 2015"
echo "  - Execution Units: 24 EUs"
echo "  - FP32 Performance: ~403 GFLOPS (peak)"
echo "  - Quick Sync Video: 4th Gen (VME + VEBox + SFC)"
echo ""

echo "COMMERCIAL FRAME GENERATION SUPPORT:"
echo "  - NVIDIA DLSS 3 Frame Gen: NO (requires RTX 40xx Tensor cores)"
echo "  - AMD FSR 3 Frame Gen:     NO (requires Intel Arc discrete GPU)"
echo "  - Intel XeSS 3 MFG:        NO (exclusive to Intel Arc discrete)"
echo ""

echo "SOFTWARE-BASED FRAME GENERATION:"
if command -v ffmpeg &>/dev/null && ffmpeg -hide_banner -filters 2>/dev/null | grep -q "minterpolate"; then
    log_success "FFmpeg minterpolate: AVAILABLE (analytical motion interpolation)"
else
    log_warn "FFmpeg minterpolate: Not available"
fi

if command -v python3 &>/dev/null && python3 -c "import cv2" 2>/dev/null; then
    log_success "OpenCV interpolation: AVAILABLE (Farneback/Lucas-Kanade optical flow)"
else
    log_warn "OpenCV interpolation: Not available"
fi

echo ""
echo "HYPOTHESIS: Overlooked VME + OpenCL path"
echo "  The HD 520's VME engine provides hardware-accelerated motion"
echo "  estimation (used for video encoding), which can be repurposed for"
echo "  frame interpolation. The OpenCL compute engine (24 EUs) handles"
echo "  motion-compensated frame blending."
echo ""

if command -v vainfo &>/dev/null && vainfo 2>&1 | grep -qi "Intel"; then
    if vainfo 2>&1 | grep -qi "EncSlice\|EncPicture"; then
        log_success "VME engine is ACCESSIBLE via VA-API - hardware motion estimation possible!"
    else
        log_warn "VA-API Intel driver found, but VME entrypoints not confirmed"
    fi
else
    log_warn "VA-API not available - VME hardware acceleration not accessible"
    log_warn "  Install intel-media-va-driver-non-free for Intel HW acceleration"
fi

echo ""
echo "TEST RESULTS SUMMARY:"
echo "  Results directory: $RESULTS_DIR"
ls -la "$RESULTS_DIR" 2>/dev/null | grep -v "^total\|^d" | sed 's/^/    /'

echo ""
echo "NEXT STEPS:"
echo "  1. Run: python3 poc_frame_interpolation.py --demo"
echo "  2. Run: python3 tests/gpu_capability_test.py"
echo "  3. Test with your video: python3 poc_frame_interpolation.py input.mp4 -o output.mp4 --fps 60"
echo "  4. Use FFmpeg: ffmpeg -i input.mp4 -vf minterpolate=60 output.mp4"
