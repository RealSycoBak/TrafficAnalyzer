from ultralytics import YOLO
import cv2
import time
import csv
import torch
import datetime
import ssl
import os
from torchvision import models, transforms
from PIL import Image
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# Fix SSL certificate issue for Python 3.13
ssl._create_default_https_context = ssl._create_unverified_context

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load DeepSORT tracker
tracker = DeepSort(max_age=30)

# Load ResNet50 for vehicle classification with error handling
try:
    vehicle_classifier = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    vehicle_classifier.eval()
    use_classifier = True
except Exception as e:
    print(f"Warning: Could not load ResNet50 classifier: {e}")
    print("Continuing with YOLO classification only")
    use_classifier = False

# Define image preprocessing for classification
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Electric vehicle classification function (simplified example)
def is_electric_vehicle(vehicle_img, model_output=None):
    """
    Determine if a vehicle is electric based on visual characteristics
    This is a simplified approach - in a real application, you would use a trained
    classifier specifically for distinguishing electric vs non-electric vehicles
    """
    # For this example, we'll use a heuristic based on vehicle color and shape
    # This is just a placeholder - a real implementation would use a trained model
    
    # Convert to HSV for better color analysis
    if vehicle_img is not None and vehicle_img.size > 0:
        try:
            hsv_img = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2HSV)
            
            # Simple heuristic: electric vehicles are often blue/white/silver
            # and have smooth, aerodynamic shapes
            
            # Calculate average hue
            avg_hue = np.mean(hsv_img[:, :, 0])
            avg_saturation = np.mean(hsv_img[:, :, 1])
            
            # Check for blue/teal colors (common in electric vehicles)
            is_blue_teal = (90 < avg_hue < 110) and avg_saturation > 50
            
            # Check for white/silver (common in Teslas and other EVs)
            is_white_silver = avg_saturation < 30
            
            # If using a classifier model, we could use specific classes
            # For example, certain ImageNet classes might correlate with EV shapes
            if model_output is not None:
                # This is a very rough approximation - a real system would need
                # a specialized classifier
                ev_related_classes = [817, 818, 661, 468]  # Sample ImageNet classes
                _, predicted = torch.max(model_output, 1)
                pred_class = predicted.item()
                class_hint = pred_class in ev_related_classes
            else:
                class_hint = False
                
            # Combine heuristics
            return (is_blue_teal or is_white_silver or class_hint)
            
        except Exception:
            return False
    
    return False

# Open video file
cap = cv2.VideoCapture("video.mp4")

# Get FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# Tracking for 5-minute intervals
interval_duration = 5 * 60  # 5 minutes in seconds
start_time = time.time()
current_interval = 0
tracked_vehicles = set()  # Track unique vehicles within interval
interval_data = {
    "vehicle_types": {},  # Count by type
    "electric_count": 0,
    "non_electric_count": 0,
    "total_count": 0
}

# Open CSV file for logging interval summaries
with open("traffic_interval_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Interval Start Time", 
        "Interval End Time", 
        "Total Vehicles", 
        "Electric Vehicles", 
        "Non-Electric Vehicles",
        "Cars", 
        "Trucks", 
        "Buses", 
        "Motorcycles", 
        "Other Vehicles",
        "Traffic Flow (vehicles/hour)"
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        # Check if the current 5-minute interval has completed
        elapsed_time = current_time - start_time
        interval_elapsed = elapsed_time % interval_duration
        
        # If a new interval has started, log the summary data and reset counters
        if int(elapsed_time / interval_duration) > current_interval:
            # Calculate the actual interval times
            interval_start = datetime.datetime.fromtimestamp(
                start_time + (current_interval * interval_duration)
            ).strftime("%Y-%m-%d %H:%M:%S")
            
            interval_end = datetime.datetime.fromtimestamp(
                start_time + ((current_interval + 1) * interval_duration)
            ).strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate traffic flow (vehicles per hour)
            vehicles_per_hour = (interval_data["total_count"] / interval_duration) * 3600
            
            # Extract vehicle type counts with default 0 for missing types
            cars = interval_data["vehicle_types"].get("car", 0)
            trucks = interval_data["vehicle_types"].get("truck", 0)
            buses = interval_data["vehicle_types"].get("bus", 0)
            motorcycles = interval_data["vehicle_types"].get("motorcycle", 0)
            other_vehicles = interval_data["total_count"] - (cars + trucks + buses + motorcycles)
            
            # Write the interval summary
            writer.writerow([
                interval_start,
                interval_end,
                interval_data["total_count"],
                interval_data["electric_count"],
                interval_data["non_electric_count"],
                cars,
                trucks,
                buses,
                motorcycles,
                other_vehicles,
                round(vehicles_per_hour, 1)
            ])
            
            # Reset for next interval
            current_interval += 1
            tracked_vehicles.clear()
            interval_data = {
                "vehicle_types": {},
                "electric_count": 0,
                "non_electric_count": 0,
                "total_count": 0
            }
            
        # Process current frame
        results = model(frame)
        detections = []

        # Extract and process detections
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Only track if confidence is decent
            if conf > 0.3:
                cls_id = int(cls)
                label = model.names[cls_id]
                
                # Only process vehicle classes
                if label in ['car', 'truck', 'bus', 'motorcycle']:
                    detections.append(((x1, y1, x2, y2), conf.item(), cls_id))

        # Update tracker with new detections
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                ltrb = track.to_ltrb()
                
                # Only process each unique vehicle once per interval
                if track_id in tracked_vehicles:
                    continue
                    
                # Ensure all values are integers
                x1, y1, x2, y2 = map(int, ltrb)
                
                # Get vehicle type
                vehicle_type = model.names[track.det_class]
                
                # Get vehicle image for electric vehicle detection
                is_electric = False
                model_output = None
                
                if x1 >= 0 and y1 >= 0 and x2 < frame.shape[1] and y2 < frame.shape[0] and x2 > x1 and y2 > y1:
                    vehicle_img = frame[y1:y2, x1:x2]
                    
                    # Process classification if ResNet is available
                    if use_classifier and vehicle_img.size > 0 and vehicle_img.shape[0] > 20 and vehicle_img.shape[1] > 20:
                        try:
                            vehicle_img_pil = Image.fromarray(cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB))
                            vehicle_tensor = transform(vehicle_img_pil).unsqueeze(0)

                            with torch.no_grad():
                                model_output = vehicle_classifier(vehicle_tensor)
                        except Exception:
                            pass
                            
                    # Determine if the vehicle is electric
                    is_electric = is_electric_vehicle(vehicle_img, model_output)
                
                # Update interval statistics
                tracked_vehicles.add(track_id)
                interval_data["total_count"] += 1
                
                # Update vehicle type counts
                interval_data["vehicle_types"][vehicle_type] = interval_data["vehicle_types"].get(vehicle_type, 0) + 1
                
                # Update electric/non-electric counts
                if is_electric:
                    interval_data["electric_count"] += 1
                else:
                    interval_data["non_electric_count"] += 1
                
                # Display on video
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Color-code electric vehicles differently
                label_color = (0, 0, 255) if is_electric else (0, 255, 0)
                ev_label = "Electric" if is_electric else "Non-electric"
                
                if y1 >= 25:
                    cv2.putText(frame, f"{vehicle_type} - {ev_label}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
                else:
                    cv2.putText(frame, f"{vehicle_type} - {ev_label}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

        # Display current interval stats
        cv2.putText(frame, f"Interval: {current_interval+1}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {int(interval_elapsed)}s / {interval_duration}s", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {interval_data['total_count']}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Electric: {interval_data['electric_count']}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Non-Electric: {interval_data['non_electric_count']}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Traffic Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Write final interval data if any vehicles were tracked
    if interval_data["total_count"] > 0:
        interval_start = datetime.datetime.fromtimestamp(
            start_time + (current_interval * interval_duration)
        ).strftime("%Y-%m-%d %H:%M:%S")
        
        interval_end = datetime.datetime.fromtimestamp(
            start_time + (current_interval * interval_duration) + elapsed_time
        ).strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate traffic flow (vehicles per hour)
        if interval_elapsed > 0:  # Avoid division by zero
            vehicles_per_hour = (interval_data["total_count"] / interval_elapsed) * 3600
        else:
            vehicles_per_hour = 0
            
        # Extract vehicle type counts
        cars = interval_data["vehicle_types"].get("car", 0)
        trucks = interval_data["vehicle_types"].get("truck", 0)
        buses = interval_data["vehicle_types"].get("bus", 0)
        motorcycles = interval_data["vehicle_types"].get("motorcycle", 0)
        other_vehicles = interval_data["total_count"] - (cars + trucks + buses + motorcycles)
        
        # Write the final interval summary
        writer.writerow([
            interval_start,
            interval_end,
            interval_data["total_count"],
            interval_data["electric_count"],
            interval_data["non_electric_count"],
            cars,
            trucks,
            buses,
            motorcycles,
            other_vehicles,
            round(vehicles_per_hour, 1)
        ])

cap.release()
cv2.destroyAllWindows()