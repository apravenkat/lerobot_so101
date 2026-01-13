import cv2
import zmq
import struct
import time

class PerceptionPublisher:
    def __init__(self, device="/dev/video2", port=5555, fps_limit=30):
        # Camera setup with aggressive buffer control
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)  # Use V4L2 backend for Linux
        
        # Set resolution BEFORE other settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # CRITICAL USB camera settings
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimum buffer
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Disable auto-exposure/white-balance to reduce processing delays
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        
        # For USB cameras: Use MJPEG if available (reduces CPU load)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        
        # Verify settings
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_buffer = self.cap.get(cv2.CAP_PROP_BUFFERSIZE)
        print(f"Camera FPS: {actual_fps}, Buffer: {actual_buffer}")
        
        # ZMQ setup
        self.context = zmq.Context()
        self.pub = self.context.socket(zmq.PUB)
        
        self.pub.setsockopt(zmq.SNDHWM, 1)
        self.pub.setsockopt(zmq.SNDBUF, 2 * 1024 * 1024)
        self.pub.setsockopt(zmq.CONFLATE, 1)
        
        self.pub.bind(f"tcp://*:{port}")
        
        # Frame markers
        self.SOF = b'\xDE\xAD\xBE\xEF'
        self.EOF = b'\xFE\xED\xFA\xCE'
        
        # FPS limiting
        self.fps_limit = fps_limit
        self.frame_interval = 1.0 / fps_limit if fps_limit > 0 else 0
        
        time.sleep(0.5)
        
        print(f"Publisher started on port {port}")
    
    def _flush_camera_buffer(self, num_frames=10):
        """Aggressively flush USB camera's internal buffer"""
        for _ in range(num_frames):
            self.cap.grab()  # grab() is faster than read()
    
    def run(self):
        frame_count = 0
        last_time = time.time()
        
        # Initial aggressive flush
        print("Flushing camera buffer...")
        self._flush_camera_buffer(30)
        
        while True:
            # CRITICAL: Always flush before reading to get the FRESHEST frame
            # USB cameras often have 3-5 frames buffered internally
            self._flush_camera_buffer(3)
            
            # FPS limiting
            current_time = time.time()
            if self.frame_interval > 0:
                elapsed = current_time - last_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                    current_time = time.time()
            
            last_time = current_time
            
            # Now read the fresh frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            # Add timestamp to verify frame freshness
            timestamp_ms = int(time.time() * 1000)
            cv2.putText(frame, f"T:{timestamp_ms}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Compress frame
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, encoded = cv2.imencode(".jpg", frame, encode_param)
            data = encoded.tobytes()
            
            # Build atomic packet with timestamp
            size_bytes = struct.pack("I", len(data))
            timestamp_bytes = struct.pack("Q", timestamp_ms)  # 8 bytes
            packet = self.SOF + size_bytes + timestamp_bytes + data + self.EOF
            
            # Send frame
            try:
                self.pub.send(packet, flags=zmq.NOBLOCK)
                frame_count += 1
                
                if frame_count % 30 == 0:
                    print(f"Sent {frame_count} frames")
                    
            except zmq.Again:
                continue
    
    def close(self):
        self.cap.release()
        self.pub.close()
        self.context.term()

if __name__ == "__main__":
    publisher = PerceptionPublisher(device="/dev/video2", port=5555, fps_limit=30)
    try:
        publisher.run()
    except KeyboardInterrupt:
        print("\nStopping publisher...")
    finally:
        publisher.close()