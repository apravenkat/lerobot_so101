import zmq
import cv2
import numpy as np

class PerceptionSubscriber:
    def __init__(self, host="localhost", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host}:{port}")

        # Subscribe to all messages (CORRECT way)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # Always keep only the latest frame (real-time safe)
        self.socket.setsockopt(zmq.CONFLATE, 1)

    def get_frame(self):
        try:
            data = self.socket.recv(flags=zmq.NOBLOCK)

            frame = cv2.imdecode(
                np.frombuffer(data, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            cv2.imshow("Received Frame", frame)
            cv2.waitKey(1)
            return frame

        except zmq.Again:
            return None

    def close(self):
        self.socket.close()
        self.context.term()
