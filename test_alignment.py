#!/usr/bin/env python3
"""
Test script to verify RGB/depth alignment.
Run this script to test the alignment endpoints and verify pixel-perfect registration.
"""

import requests
import cv2
import numpy as np
import time
from typing import Optional

class AlignmentTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    def test_endpoints(self):
        """Test all alignment-related endpoints."""
        print("Testing RGB/Depth Alignment Endpoints")
        print("=" * 50)
        
        # Test verification endpoint
        try:
            response = requests.get(f"{self.base_url}/stream/verify-alignment")
            if response.status_code == 200:
                data = response.json()
                print("Alignment Verification Metrics:")
                print(f"  Correlation: {data.get('correlation', 'N/A'):.3f}")
                print(f"  RGB Edges: {data.get('rgb_edges', 'N/A')}")
                print(f"  Depth Edges: {data.get('depth_edges', 'N/A')}")
                print(f"  Overlap Edges: {data.get('overlap_edges', 'N/A')}")
                print(f"  Overlap Percentage: {data.get('overlap_percentage', 'N/A'):.1f}%")
                
                # Interpret results
                correlation = data.get('correlation', 0)
                overlap_pct = data.get('overlap_percentage', 0)
                
                if correlation > 0.7 and overlap_pct > 60:
                    print("✅ EXCELLENT alignment - RGB and depth are well registered")
                elif correlation > 0.5 and overlap_pct > 40:
                    print("✅ GOOD alignment - RGB and depth are reasonably registered")
                elif correlation > 0.3 and overlap_pct > 20:
                    print("⚠️  FAIR alignment - Some registration issues detected")
                else:
                    print("❌ POOR alignment - Significant registration problems")
                    
            else:
                print(f"❌ Failed to get verification data: {response.status_code}")
        except Exception as e:
            print(f"❌ Error testing verification endpoint: {e}")
            
        print("\nStreaming Endpoints Available:")
        print(f"  RGB Stream: {self.base_url}/stream/rgb")
        print(f"  Depth Stream: {self.base_url}/stream/depth")
        print(f"  Aligned Stream: {self.base_url}/stream/aligned")
        print(f"  Verification: {self.base_url}/stream/verify-alignment")
        
    def capture_alignment_image(self, save_path: str = "alignment_test.jpg"):
        """Capture and save the current alignment visualization."""
        try:
            response = requests.get(f"{self.base_url}/stream/aligned", stream=True)
            if response.status_code == 200:
                # Read the first frame from the MJPEG stream
                frame_data = b""
                for chunk in response.iter_content(chunk_size=1024):
                    frame_data += chunk
                    if b"\r\n\r\n" in frame_data:
                        # Find the end of the first frame
                        frame_end = frame_data.find(b"\r\n\r\n") + 4
                        frame_data = frame_data[frame_end:]
                        break
                
                # Save the frame
                with open(save_path, 'wb') as f:
                    f.write(frame_data)
                print(f"✅ Alignment image saved to {save_path}")
                return True
            else:
                print(f"❌ Failed to capture alignment image: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error capturing alignment image: {e}")
            return False

def main():
    """Main test function."""
    print("RGB/Depth Alignment Test")
    print("=" * 30)
    print("Make sure your Kinect server is running on localhost:8000")
    print()
    
    tester = AlignmentTester()
    
    # Test endpoints
    tester.test_endpoints()
    
    print("\n" + "=" * 50)
    print("Manual Verification Steps:")
    print("1. Open your browser and go to:")
    print("   - http://localhost:8000/stream/rgb (RGB stream)")
    print("   - http://localhost:8000/stream/depth (Depth stream)")
    print("   - http://localhost:8000/stream/aligned (Overlaid stream)")
    print()
    print("2. Compare the streams:")
    print("   - RGB and depth should show the same objects")
    print("   - Edges and features should align pixel-perfectly")
    print("   - The aligned stream should show good overlay")
    print()
    print("3. Check the verification metrics:")
    print("   - Correlation should be > 0.5 for good alignment")
    print("   - Overlap percentage should be > 40% for good alignment")
    print()
    print("4. If alignment is poor:")
    print("   - Check that freenect registration is working")
    print("   - Verify Kinect is properly connected")
    print("   - Ensure good lighting conditions")
    
    # Try to capture an alignment image
    print("\nCapturing alignment test image...")
    tester.capture_alignment_image()

if __name__ == "__main__":
    main()
