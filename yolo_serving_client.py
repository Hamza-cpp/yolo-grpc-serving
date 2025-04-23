import grpc
import logging

import yolo_serving_pb2
import yolo_serving_pb2_grpc

# --- Configuration ---
CLIENT_CONFIG = {
    "server_address": "localhost:50051",
    "default_confidence_threshold": 0.1,
    "default_timeout": 10,  # seconds
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
            request = yolo_serving_pb2.HealthCheckRequest()
            response = self.stub.CheckHealth(request, timeout=timeout)
            return response.status
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            return yolo_serving_pb2.HealthCheckResponse.ServingStatus.UNKNOWN


def display_detections(response):
    """Displays detection results in a formatted way."""
    logging.info(f"Received {len(response.boxes)} detections:")
    for i, box in enumerate(response.boxes):
        print(f"  Box {i + 1}:")
        print(f"    Class ID: {box.class_id}")
        print(f"    Class Name: {box.class_name}")
        print(f"    Confidence: {box.confidence:.4f}")
        print(
            f"    Coords (ymin, xmin, ymax, xmax): "
            f"({box.y_min:.2f}, {box.x_min:.2f}, {box.y_max:.2f}, {box.x_max:.2f})"
        )


def run():
    """Main function to run the client."""
    logging.basicConfig(level=logging.INFO)

    image_path = "../Downloads/test-image.jpeg"

    try:
        # Using context manager to handle channel lifecycle
        with YoloClient() as client:
            # Check server health
            logging.info("Checking server health...")
            health_status = client.check_health()
            logging.info(f"Server health status: {health_status}")
            if (
                health_status
                != yolo_serving_pb2.HealthCheckResponse.ServingStatus.SERVING
            ):
                logging.error("Server is not healthy!")
                return

            # Send prediction request
            response = client.predict(image_path)

            # Display results
            display_detections(response)

    except Exception as e:
        logging.error(f"Client execution failed: {e}")


if __name__ == "__main__":
    run()
