"""
Tunisian Currency Detection with YOLOv11
Real-time detection of Tunisian Dinar banknotes (1, 5, 10, 20, 50 DT)
Author: Ahmed Omar Dridi
"""

from ultralytics import YOLO
import cv2
import time

# Map class IDs to currency values
# Model has 4 classes: class_0, class_1, class_2, class_3
# Mapping based on testing:
CLASS_TO_VALUE = {
    0: 10,   # class_0 -> 10 DT (confirmed: blue bill)
    1: 20,   # class_1 -> 20 DT (confirmed)
    2: 5,    # class_2 -> 5 DT
    3: 50,   # class_3 -> 50 DT
}

CLASS_TO_NAME = {
    0: '10 DT',
    1: '20 DT',
    2: '5 DT',
    3: '50 DT',
}

def main():
    # Load the trained model
    model = YOLO("best.pt")
    
    # Print model classes for verification
    print("Model classes:", model.names)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    print("=" * 50)
    print("Tunisian Currency Detection - YOLOv11")
    print("=" * 50)
    print("Press 'q' to quit")
    print("=" * 50)
    
    # Timer for console output
    last_time = time.time()
    interval = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break
        
        # Run detection
        results = model(frame, conf=0.5)
        
        # Calculate total
        total = 0
        bill_counts = {}
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.cpu().numpy()[0])
                
                # Get value from mapping
                value = CLASS_TO_VALUE.get(class_id, 0)
                name = CLASS_TO_NAME.get(class_id, f'class_{class_id}')
                
                total += value
                
                if name not in bill_counts:
                    bill_counts[name] = {'count': 0, 'value': value}
                bill_counts[name]['count'] += 1
        
        # Print to console periodically
        current_time = time.time()
        if current_time - last_time >= interval:
            print(f"\n[{time.strftime('%H:%M:%S')}] Detection Results:")
            print(f"  Bills detected: {sum(b['count'] for b in bill_counts.values())}")
            for name, data in bill_counts.items():
                print(f"    - {name}: {data['count']}x = {data['count'] * data['value']} DT")
            print(f"  TOTAL: {total} DT")
            print("-" * 30)
            last_time = current_time
        
        # Annotate frame with custom labels
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
                xyxy = box.xyxy.cpu().numpy()[0].astype(int)
                
                value = CLASS_TO_VALUE.get(class_id, 0)
                name = CLASS_TO_NAME.get(class_id, f'class_{class_id}')
                
                # Draw box
                cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 255), 3)
                
                # Draw label
                label = f"{name} ({conf*100:.0f}%)"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]-30), (xyxy[0]+w+10, xyxy[1]), (0, 255, 255), -1)
                cv2.putText(annotated_frame, label, (xyxy[0]+5, xyxy[1]-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add total overlay
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        cv2.putText(annotated_frame, f"TOTAL: {total} DT", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
        
        num_bills = sum(b['count'] for b in bill_counts.values())
        cv2.putText(annotated_frame, f"Bills: {num_bills}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow("Tunisian Currency Detection - YOLOv11", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDetection stopped.")

if __name__ == "__main__":
    main()
