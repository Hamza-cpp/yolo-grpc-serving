import grpc
import logging
import os
import sys
import argparse

import yolo_serving_pb2
import yolo_serving_pb2_grpc

# --- Configuration ---
CLIENT_CONFIG = {
    "server_address": os.environ.get("SERVER_ADDRESS", "localhost:50051"),
    "default_confidence_threshold": float(os.environ.get("CONFIDENCE_THRESHOLD", "0.1")),
    "default_timeout": int(os.environ.get("REQUEST_TIMEOUT", "10")),
    "test_image": os.environ.get("TEST_IMAGE", "./images/test-image.jpeg")
}


class YoloClient:
    """Client for interacting with YOLO gRPC service."""

    def __init__(self, server_address=None, use_secure=False, cert_path=None):
        """Initialize the client with connection settings."""
        self.server_address = server_address or CLIENT_CONFIG["server_address"]
        self.use_secure = use_secure
        self.cert_path = cert_path
        self.channel = None
        self.stub = None

        logging.info(f"Initializing client to connect to server at {self.server_address}")

    def __enter__(self):
        """Context manager entry method that creates channel."""
        self._create_channel()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method that closes channel."""
        if self.channel:
            self.channel.close()
            self.channel = None
            self.stub = None

    def _create_channel(self):
        """Creates and initializes the gRPC channel."""
        if self.use_secure and self.cert_path:
            credentials = grpc.ssl_channel_credentials(
                root_certificates=open(self.cert_path, "rb").read()
            )
            self.channel = grpc.secure_channel(self.server_address, credentials)
        else:
            self.channel = grpc.insecure_channel(self.server_address)

        self.stub = yolo_serving_pb2_grpc.YoloServiceStub(self.channel)

    def predict(self, image_path, confidence_threshold=None, timeout=None):
        """Sends prediction request to the server."""
        if not self.stub:
            self._create_channel()

        conf_threshold = (
            confidence_threshold or CLIENT_CONFIG["default_confidence_threshold"]
        )
        req_timeout = timeout or CLIENT_CONFIG["default_timeout"]

        logging.info(f"Reading image: {image_path}")
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                logging.error(f"Image file not found: {image_path}")
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Read image
            with open(image_path, "rb") as f:
                image_bytes = f.read()

            # Create request
            request = yolo_serving_pb2.YoloRequest(
                image_data=image_bytes, confidence_threshold=conf_threshold
            )

            # Send request
            logging.info("Sending prediction request...")
            response = self.stub.Predict(request, timeout=req_timeout)

            return response

        except grpc.RpcError as e:
            logging.error(f"gRPC call failed: {e.code()} - {e.details()}")
            raise
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

    def check_health(self, timeout=3):
        """Checks the health status of the server."""
        if not self.stub:
            self._create_channel()

        try:
            logging.info(f"Checking health of server at {self.server_address}")
            request = yolo_serving_pb2.HealthCheckRequest()
            response = self.stub.CheckHealth(request, timeout=timeout)
            
            status_map = {
                0: "UNKNOWN",
                1: "SERVING",
                2: "NOT_SERVING",
                3: "SERVICE_UNKNOWN"
            }
            status_str = status_map.get(response.status, f"INVALID({response.status})")
            logging.info(f"Server health status: {status_str}")
            
            return response.status
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return yolo_serving_pb2.HealthCheckResponse.ServingStatus.UNKNOWN


def display_detections(response):
    """Displays detection results in a formatted way."""
    box_count = len(response.boxes)
    logging.info(f"Received {box_count} detection{'s' if box_count != 1 else ''}:")
    
    if box_count == 0:
        print("  No objects detected.")
        return
    
    for i, box in enumerate(response.boxes):
        print(f"  Box {i + 1}:")
        print(f"    Class ID: {box.class_id}")
        print(f"    Class Name: {box.class_name}")
        print(f"    Confidence: {box.confidence:.4f}")
        print(
            f"    Coords (ymin, xmin, ymax, xmax): "
            f"({box.y_min:.2f}, {box.x_min:.2f}, {box.y_max:.2f}, {box.x_max:.2f})"
        )

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO gRPC Client")
    parser.add_argument(
        "--server", 
        default=CLIENT_CONFIG["server_address"],
        help="Server address in format host:port"
    )
    parser.add_argument(
        "--image", 
        default=CLIENT_CONFIG["test_image"],
        help="Path to image file for inference"
    )
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=CLIENT_CONFIG["default_confidence_threshold"],
        help="Confidence threshold for detections"
    )
    parser.add_argument(
        "--check-health", 
        action="store_true",
        help="Only check server health without sending prediction request"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()

def run():
    """Main function to run the client."""
    args= parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=log_level, format=log_format)
    
    logging.info(f"YOLO gRPC client starting")
    logging.info(f"Server address: {args.server}")

    try:
        # Using context manager to handle channel lifecycle
        with YoloClient(server_address=args.server) as client:
            # Check server health
            health_status = client.check_health()
            
            # If health check mode, exit after check
            if args.check_health:
                sys.exit(0 if health_status == yolo_serving_pb2.HealthCheckResponse.ServingStatus.SERVING else 1)

            # If server not healthy, exit
            if health_status != yolo_serving_pb2.HealthCheckResponse.ServingStatus.SERVING:
                logging.error("Server is not healthy! Exiting.")
                sys.exit(1)

            # Send prediction request
            logging.info(f"Processing image: {args.image}")
            response = client.predict(args.image, confidence_threshold=args.confidence)


            # Display results
            display_detections(response)
            logging.info("Prediction completed successfully.")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(1)
    except grpc.RpcError as e:
        logging.error(f"gRPC error: [{e.code()}] {e.details()}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Client execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
