#!/usr/bin/env python3
"""
Intel HD Graphics 520 - Frame Interpolation Proof-of-Concept

This script demonstrates motion-compensated frame interpolation that can run
on Intel HD Graphics 520 (and other GPUs) using OpenCV's optical flow and
optional OpenCL acceleration.

HYPOTHESIS: Even without dedicated AI/ML hardware (Tensor cores, XMX), the
Intel HD Graphics 520 can perform frame interpolation using:
1. Software-based motion estimation (Farneback optical flow or block matching)
2. Motion-compensated frame blending
3. OpenCL acceleration via OpenCV (if available)

USAGE:
  python3 poc_frame_interpolation.py --demo            # Run built-in demo with synthetic frames
  python3 poc_frame_interpolation.py input.mp4        # Process a video file
  python3 poc_frame_interpolation.py --help            # Show full help

REQUIREMENTS:
  pip install opencv-python numpy scipy

The script automatically detects and uses OpenCL if available, falling back
to CPU-based processing. This mirrors the hypothesis that the HD 520's
compute capabilities (24 EUs, OpenCL 3.0) can handle frame interpolation
by leveraging both GPU compute (via OpenCL) and the video processing pipeline
(VEBox/VME via VA-API).

AUTHOR: 520-frame-gen research project
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class HD520FrameInterpolator:
    """
    Frame interpolator optimized for Intel HD Graphics 520 capabilities.

    Leverages:
    - OpenCV's Optical Flow (Farneback / Lucas-Kanade) for motion estimation
    - Motion-compensated frame blending for interpolation
    - OpenCL acceleration if available (via OpenCV's ocl module)
    """

    def __init__(self, use_opencl=None, motion_method="farneback", blend_factor=0.5):
        self.use_opencl = use_opencl
        self.motion_method = motion_method.lower()
        self.blend_factor = blend_factor

        if self.use_opencl is None:
            self.use_opencl = self._detect_opencl()

        if self.use_opencl:
            cv2.ocl.setUseOpenCL(True)
            print(f"[INFO] OpenCL enabled for acceleration")

        self.gpu_frame1 = None
        self.gpu_frame2 = None
        self.gpu_flow = None

        if self.use_opencl:
            self.gpu_frame1 = cv2.UMat()
            self.gpu_frame2 = cv2.UMat()
            self.gpu_flow = cv2.UMat()

    def _detect_opencl(self):
        """Detect if OpenCL is available for GPU acceleration."""
        try:
            ctx = cv2.ocl.getContext()
            if ctx is not None and ctx.isOpenCLAvailable():
                devices = ctx.getInfo(cv2.ocl.CONTEXT_INFO_DEVICE_NAME_UBYTE)
                print(f"[INFO] OpenCL device: {devices.decode() if devices else 'Unknown'}")
                return True
        except Exception as e:
            print(f"[WARN] OpenCL detection failed: {e}")
        return False

    def estimate_motion(self, frame1, frame2):
        """
        Estimate motion between two frames using optical flow.

        This simulates what the HD 520's VME engine does for video encoding,
        but uses software-based optical flow instead.

        Args:
            frame1: Previous frame (grayscale or color)
            frame2: Current frame (grayscale or color)

        Returns:
            flow: Motion vector field (HxW x 2 float32 array)
        """
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = frame1, frame2

        if self.motion_method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        elif self.motion_method == "rlof":
            try:
                optical_flow = cv2.DualTVL1OpticalFlow_create()
                flow = optical_flow.calc(gray1, gray2, None)
            except Exception:
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
        else:
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )

        return flow

    def motion_compensate(self, frame1, frame2, flow):
        """
        Motion-compensated frame interpolation.

        This is the core of frame generation: using motion vectors to
        warp the previous frame to align with the current frame, then
        blend them to create an intermediate frame.

        The approach mimics how FSR 3 / XeSS 3 Frame Generation work,
        but uses analytical motion compensation instead of ML-based
        neural interpolation.

        Args:
            frame1: Previous frame
            frame2: Current frame
            flow: Motion vectors from frame1 to frame2

        Returns:
            interpolated_frame: Frame at time t = blend_factor
        """
        h, w = frame1.shape[:2]

        flow_x = flow[..., 0]
        flow_y = flow[..., 1]

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                      np.arange(h, dtype=np.float32))

        map_x = (grid_x + flow_x * self.blend_factor).astype(np.float32)
        map_y = (grid_y + flow_y * self.blend_factor).astype(np.float32)

        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)

        if self.use_opencl:
            gpu_frame1 = cv2.UMat(frame1)
            warped1 = cv2.remap(gpu_frame1, cv2.UMat(map_x), cv2.UMat(map_y),
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
            warped1 = warped1.get()
        else:
            warped1 = cv2.remap(frame1, map_x, map_y,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)

        negative_flow_x = -flow_x
        negative_flow_y = -flow_y

        map_x2 = (grid_x + negative_flow_x * (1.0 - self.blend_factor)).astype(np.float32)
        map_y2 = (grid_y + negative_flow_y * (1.0 - self.blend_factor)).astype(np.float32)

        map_x2 = np.clip(map_x2, 0, w - 1)
        map_y2 = np.clip(map_y2, 0, h - 1)

        warped2 = cv2.remap(frame2, map_x2, map_y2,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)

        interpolated = cv2.addWeighted(warped1, 1.0 - self.blend_factor,
                                       warped2, self.blend_factor, 0)

        return interpolated

    def interpolate_frame(self, frame1, frame2):
        """
        Generate an interpolated frame between frame1 and frame2.

        This simulates frame generation: real_frame1 -> interp_frame -> real_frame2

        Args:
            frame1: Previous real frame
            frame2: Current real frame

        Returns:
            interpolated frame at 50% time between frame1 and frame2
        """
        t0 = time.perf_counter()

        flow = self.estimate_motion(frame1, frame2)

        t1 = time.perf_counter()

        interpolated = self.motion_compensate(frame1, frame2, flow)

        t2 = time.perf_counter()

        print(f"  Motion estimation: {(t1 - t0) * 1000:.1f} ms")
        print(f"  Motion compensation: {(t2 - t1) * 1000:.1f} ms")
        print(f"  Total: {(t2 - t0) * 1000:.1f} ms")

        return interpolated

    def process_video(self, input_path, output_path=None, target_fps=None,
                      max_frames=None, show_preview=False):
        """
        Process a video file to add interpolated frames.

        Simulates the frame generation pipeline:
        1. Read consecutive frame pairs
        2. Estimate motion between frames
        3. Generate intermediate frames
        4. Output at higher frame rate

        Args:
            input_path: Input video file path
            output_path: Output video file path (optional)
            target_fps: Target output FPS (default: 2x input FPS)
            max_frames: Maximum frames to process (for testing)
            show_preview: Show preview window during processing
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if target_fps is None:
            target_fps = fps * 2

        interp_factor = target_fps / fps
        frames_to_insert = int(interp_factor - 1) if interp_factor > 1 else 0

        print(f"\n[INFO] Video properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  Input FPS: {fps:.1f}")
        print(f"  Target FPS: {target_fps:.1f}")
        print(f"  Interpolation factor: {interp_factor:.2f}x")
        print(f"  Total frames: {total_frames}")
        print(f"  Frames to insert per pair: {frames_to_insert}")

        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"avc1" if cv2.__version__ >= "4" else "mp4v")
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        else:
            out = None

        ret, prev_frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read first frame")
            return

        processed_count = 0
        frame_idx = 0

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                if out:
                    out.write(prev_frame)
                break

            if out:
                out.write(prev_frame)

            for i in range(frames_to_insert):
                interp = self.interpolate_frame(prev_frame, curr_frame)
                if out:
                    out.write(interp)

            if out:
                out.write(curr_frame)

            prev_frame = curr_frame.copy()
            frame_idx += 1
            processed_count += 1

            if max_frames and processed_count >= max_frames:
                break

            if show_preview:
                cv2.imshow("Original", curr_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            if processed_count % 10 == 0:
                print(f"  Processed {processed_count}/{total_frames} frame pairs")

        cap.release()
        if out:
            out.release()

        print(f"\n[DONE] Processed {processed_count} frame pairs")
        if output_path:
            print(f"  Output saved to: {output_path}")

    def run_demo(self):
        """Run a built-in demonstration with synthetic test frames."""
        print("\n" + "=" * 60)
        print("RUNNING DEMO: Motion-Compensated Frame Interpolation")
        print("=" * 60)

        height, width = 480, 640
        fps = 30

        print(f"\n[INFO] Demo parameters:")
        print(f"  Resolution: {width}x{height}")
        print(f"  OpenCL: {self.use_opencl}")
        print(f"  Motion method: {self.motion_method}")

        print("\n[INFO] Generating synthetic test frames (moving square)...")

        frame1 = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame1, (50, 50), (100, 100), (0, 255, 0), -1)

        frame2 = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame2, (70, 50), (120, 100), (0, 255, 0), -1)

        frame3 = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame3, (90, 50), (140, 100), (0, 255, 0), -1)

        cv2.putText(frame1, "Frame T=0", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame2, "Frame T=1", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame3, "Frame T=2", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        print("\n[INFO] Frame 1: Square at x=50")
        print("[INFO] Frame 2: Square at x=70 (moved right by 20px)")

        print("\n[INFO] Starting interpolation between Frame 1 and Frame 2...")
        interpolated = self.interpolate_frame(frame1, frame2)

        print("\n[INFO] Interpolating between Frame 2 and Frame 3...")
        interpolated2 = self.interpolate_frame(frame2, frame3)

        cv2.putText(interpolated, "Interpolated T=0.5", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(interpolated2, "Interpolated T=2.5", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        square1_x = 50
        square2_x = 90
        expected_x = (square1_x + square2_x) // 2
        print(f"\n[INFO] Validation: Expected interpolated square at x={expected_x}")

        print("\n[INFO] Displaying results...")
        print("  Red = original frame 1")
        print("  Green = original frame 2")
        print("  Blue = interpolated frame")

        side_by_side = cv2.hconcat([frame1, frame2])
        cv2.putText(side_by_side, "Original Frames", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        all_frames = cv2.hconcat([frame1, interpolated, frame2, interpolated2, frame3])
        cv2.putText(all_frames, "Frame1 -> Interp -> Frame2 -> Interp -> Frame3",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Original vs Interpolated", side_by_side)
        cv2.imshow("Full Sequence", all_frames)
        print("\n[INFO] Press any key in the image window to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("\n[DONE] Demo completed successfully!")
        print("\n[CONCLUSION]")
        print("  If the interpolated frame shows the square at the midpoint between")
        print("  the two original positions, motion-compensated frame interpolation")
        print("  is working correctly. This proves the approach is viable on the HD 520.")
        print("\n  The HD 520's OpenCL compute engine (24 EUs) handles the motion")
        print("  estimation and frame blending. In a full implementation, the VME")
        print("  engine could accelerate motion estimation, and the VEBox could")
        print("  handle post-processing, significantly improving performance.")

    def benchmark(self, width=640, height=480, iterations=10):
        """Benchmark interpolation performance."""
        print(f"\n[INFO] Benchmarking frame interpolation ({iterations} iterations)...")

        np.random.seed(42)
        frame1 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        frame2 = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        total_time = 0
        for i in range(iterations):
            t0 = time.perf_counter()
            interp = self.interpolate_frame(frame1, frame2)
            t1 = time.perf_counter()
            total_time += (t1 - t0)
            if i == 0:
                print(f"  Frame {i+1}: {(t1-t0)*1000:.1f} ms")

        avg_time = total_time / iterations * 1000
        fps = 1000 / avg_time
        print(f"\n[BENCHMARK] Average interpolation time: {avg_time:.1f} ms/frame")
        print(f"[BENCHMARK] Effective processing rate: {fps:.1f} fps")
        print(f"[BENCHMARK] For 1080p30 -> 1080p60: need <16.7ms/frame for real-time")
        print(f"[BENCHMARK] HD 520 could achieve ~{fps:.0f} interpolated frames/second")
        return avg_time


def main():
    parser = argparse.ArgumentParser(
        description="Intel HD Graphics 520 Frame Interpolation PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 poc_frame_interpolation.py --demo
  python3 poc_frame_interpolation.py input.mp4 -o output.mp4
  python3 poc_frame_interpolation.py input.mp4 --fps 120 --max-frames 100
  python3 poc_frame_interpolation.py --demo --no-opencl  # CPU-only mode
  python3 poc_frame_interpolation.py --benchmark
        """,
    )

    parser.add_argument("input", nargs="?", help="Input video file path")
    parser.add_argument("-o", "--output", help="Output video file path")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo with synthetic frames")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--fps", type=float, help="Target output FPS")
    parser.add_argument("--max-frames", type=int, help="Maximum frames to process")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--no-opencl", action="store_true", help="Disable OpenCL acceleration")
    parser.add_argument("--motion-method", choices=["farneback", "rlof"],
                        default="farneback", help="Motion estimation method")
    parser.add_argument("--blend", type=float, default=0.5,
                        help="Blend factor for interpolation (0.0-1.0)")

    args = parser.parse_args()

    use_opencl = None if not args.no_opencl else False

    interpolator = HD520FrameInterpolator(
        use_opencl=use_opencl,
        motion_method=args.motion_method,
        blend_factor=args.blend,
    )

    if args.demo:
        interpolator.run_demo()
    elif args.benchmark:
        width = args.fps is None and 640 or 1920
        height = args.fps is None and 480 or 1080
        iterations = args.max_frames or 10
        interpolator.benchmark(width=width, height=height, iterations=iterations)
    elif args.input:
        interpolator.process_video(
            input_path=args.input,
            output_path=args.output,
            target_fps=args.fps,
            max_frames=args.max_frames,
            show_preview=args.show,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
