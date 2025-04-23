import grpc
from concurrent import futures
import time
import logging
import io
import os
from PIL import Image
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

import yolo_serving_pb2
import yolo_serving_pb2_grpc

# --- Configuration ---
SERVER_CONFIG = {
    "address": f"[::]:{os.environ.get('SERVER_PORT', '50051')}",
    "max_workers": int(os.environ.get('MAX_WORKERS', '10')),
    "model_path": os.environ.get('MODEL_PATH', 'models/yolo11l.pt'),
    "default_conf_threshold": float(os.environ.get('DEFAULT_CONF_THRESHOLD', '0.1')),
    "input_img_size": int(os.environ.get('INPUT_IMG_SIZE', '640'))
}


class YoloModel:
    """Wrapper class for YOLO model functionality."""

    def __init__(self, model_path):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model(model_path)

    def _load_model(self, model_path):
        """Loads and initializes the YOLO model."""
        try:
            logging.info(f"Loading Ultralytics YOLO model from: {model_path}")
            self.model = YOLO(model_path)
            logging.info(f"Model loaded successfully. Using device: {self.device}")

            # Optional: Warm up the model
            self._warm_up()

        except Exception as e:
            logging.error(f"FATAL: Failed to load YOLO model: {e}", exc_info=True)
            raise

    def _warm_up(self):
        """Performs a warmup inference to initialize the model."""
        try:
            dummy_image = Image.new("RGB", (640, 640), color="red")
            self.model(dummy_image, verbose=False)
            logging.info("Model warm-up successful.")
        except Exception as e:
            logging.warning(f"Model warm-up failed: {e}")

    def predict(self, image, conf_threshold=None):
        """Performs inference on the given image."""
        if conf_threshold is not None:
            return self.model(image, verbose=False, conf=conf_threshold)
        else:
            return self.model(image, verbose=False)

    def is_ready(self):
        """Checks if the model is loaded and ready."""
        return self.model is not None


class ImageProcessor:
    """Handles image processing tasks."""

    @staticmethod
    def load_image(image_bytes):
        """Loads image bytes into a PIL Image object."""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_size = image.size
            return image, original_size
        except Exception as e:
            logging.error(f"Error during image loading: {e}", exc_info=True)
            raise

    @staticmethod
    def convert_results_to_boxes(result, conf_threshold):
        """Converts Ultralytics Results to gRPC BoundingBox messages."""
        boxes_list = []
        if result.boxes is None:
            logging.debug("No 'boxes' attribute found in results.")
            return boxes_list

        boxes_obj: Boxes = result.boxes
        class_names = result.names
        detections = boxes_obj.data.cpu().numpy()

        for det in detections:
            x_min, y_min, x_max, y_max = det[0:4]
            conf = float(det[4])
            class_id = int(det[5])

            if conf >= conf_threshold:
                class_name_str = class_names.get(class_id, f"class_{class_id}")

                try:
                    grpc_box = yolo_serving_pb2.BoundingBox(
                        y_min=float(y_min),
                        x_min=float(x_min),
                        y_max=float(y_max),
                        x_max=float(x_max),
                        confidence=conf,
                        class_id=class_id,
                        class_name=class_name_str,
                    )
                    boxes_list.append(grpc_box)
                except Exception as ex:
                    logging.error(
                        f"Error creating protobuf message for box {det}: {ex}"
                    )

        return boxes_list


class YoloServicer(yolo_serving_pb2_grpc.YoloServiceServicer):
    """gRPC service implementation for YOLO inference."""

    def __init__(self, model):
        self.model = model
        self.processor = ImageProcessor()

    def Predict(self, request, context):
        """Handles prediction requests using YOLO model."""
        request_id = context.peer()
        logging.info(f"Received prediction request from {request_id}")

        try:
            start_time = time.time()

            # Get confidence threshold from request or use default
            conf_threshold = (
                request.confidence_threshold
                if request.confidence_threshold > 0
                else SERVER_CONFIG["default_conf_threshold"]
            )

            # Validate request
            if not request.image_data:
                logging.warning(f"Request from {request_id} contained no image data.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Image data cannot be empty.")
                return yolo_serving_pb2.YoloResponse()

            # Process image
            prep_start = time.time()
            pil_image, _ = self.processor.load_image(request.image_data)
            prep_time = time.time() - prep_start
            logging.info(f"Request {request_id}: Image loading time: {prep_time:.4f}s")

            # Run inference
            infer_start = time.time()
            inference_results = self.model.predict(pil_image, conf_threshold)
            infer_time = time.time() - infer_start
            logging.info(f"Request {request_id}: Inference time: {infer_time:.4f}s")

            # Process results
            post_start = time.time()
            detected_boxes = []
            if (
                inference_results
                and isinstance(inference_results, list)
                and len(inference_results) > 0
            ):
                result = inference_results[0]
                detected_boxes = self.processor.convert_results_to_boxes(
                    result, conf_threshold
                )

            post_time = time.time() - post_start
            logging.info(
                f"Request {request_id}: Response formatting time: {post_time:.4f}s"
            )

            # Create response
            response = yolo_serving_pb2.YoloResponse(boxes=detected_boxes)

            total_time = time.time() - start_time
            logging.info(
                f"Prediction successful for {request_id}. Found {len(detected_boxes)} boxes. "
                f"Total time: {total_time:.4f}s"
            )

            return response

        except Exception as e:
            logging.error(f"Prediction failed for {request_id}: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error during prediction: {str(e)}")
            return yolo_serving_pb2.YoloResponse()

    def CheckHealth(self, request, context):
        """Basic health check."""
        logging.debug(f"Received health check request from {context.peer()}")
        if self.model.is_ready():
            return yolo_serving_pb2.HealthCheckResponse(
                status=yolo_serving_pb2.HealthCheckResponse.ServingStatus.SERVING
            )
        else:
            return yolo_serving_pb2.HealthCheckResponse(
                status=yolo_serving_pb2.HealthCheckResponse.ServingStatus.NOT_SERVING
            )


def serve():
    """Main function to start the server."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s",
    )

    # Initialize model
    try:
        yolo_model = YoloModel(SERVER_CONFIG["model_path"])
    except Exception as e:
        logging.error(f"Failed to initialize YOLO model: {e}")
        return

    # Create server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=SERVER_CONFIG["max_workers"])
    )
    yolo_serving_pb2_grpc.add_YoloServiceServicer_to_server(
        YoloServicer(yolo_model), server
    )

    # Start server
    server.add_insecure_port(SERVER_CONFIG["address"])
    logging.info(
        f"Starting gRPC server on {SERVER_CONFIG['address']} "
        f"with {SERVER_CONFIG['max_workers']} workers..."
    )
    server.start()
    logging.info("Server started successfully. Waiting for requests...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Stopping server...")
        server.stop(grace=5)
        logging.info("Server stopped gracefully.")


if __name__ == "__main__":
    serve()
