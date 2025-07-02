import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

yolo = YOLO('yolov8m.pt')  

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def calculate_ear(landmarks, left_idxs, right_idxs):
    def dist(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    left = [landmarks[i] for i in left_idxs]
    right = [landmarks[i] for i in right_idxs]

    left_ear = (dist(left[1], left[5]) + dist(left[2], left[4])) / (2.0 * dist(left[0], left[3]) + 1e-6)
    right_ear = (dist(right[1], right[5]) + dist(right[2], right[4])) / (2.0 * dist(right[0], right[3]) + 1e-6)

    return (left_ear + right_ear) / 2.0

EAR_THRESHOLD = 0.22
LABEL_COLORS = {
    'attentive': (0, 255, 0),
    'distracted': (0, 0, 255),
    'sleeping': (255, 0, 0),
    'using_phone': (0, 165, 255),
    'query': (255, 255, 0),
}

model_points = np.array([
    (0.0, 0.0, 0.0),            
    (0.0, -330.0, -65.0),       
    (-225.0, 170.0, -135.0),    
    (225.0, 170.0, -135.0),    
    (-150.0, -150.0, -125.0),    
    (150.0, -150.0, -125.0)      
], dtype=np.float64)

cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    label = 'attentive' 

    h, w, _ = frame.shape

    face_result = face_mesh.process(rgb)
    eyes_open = False

    yaw = 0 

    if face_result.multi_face_landmarks:
        face_landmarks = face_result.multi_face_landmarks[0].landmark
        landmarks = [(lm.x * w, lm.y * h) for lm in face_landmarks]

        left_idxs = [33, 160, 158, 133, 153, 144]
        right_idxs = [362, 385, 387, 263, 373, 380]
        ear = calculate_ear(landmarks, left_idxs, right_idxs)
        eyes_open = ear >= EAR_THRESHOLD

        image_points = np.array([
            landmarks[1],    
            landmarks[152],  
            landmarks[33],    
            landmarks[263],   
            landmarks[78],   
            landmarks[308]    
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            yaw = angles[1]  

    phone_detected = False
    results = yolo(frame)[0]
    for box in results.boxes:
        cls_id = int(box.cls)
        if yolo.names[cls_id] == 'cell phone' and box.conf > 0.5:
            phone_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), LABEL_COLORS['using_phone'], 2)
            cv2.putText(frame, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, LABEL_COLORS['using_phone'], 2)

    # === Pose Detection (Arms Raised) ===
    arms_raised = False
    pose_result = pose.process(rgb)
    if pose_result.pose_landmarks:
        wrist = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        shoulder = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        if wrist.y < shoulder.y:
            arms_raised = True

    # === Final Classification ===
    if phone_detected:
        label = 'using_phone'
    elif arms_raised:
        label = 'query'
    elif not eyes_open:
        label = 'sleeping'
    elif abs(yaw) > 40:  # Distracted only if yaw > 70 degrees
        label = 'distracted'
    else:
        label = 'attentive'

    # === Display Label ===
    color = LABEL_COLORS[label]
    cv2.putText(frame, f'{label.upper()}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Real-Time Attention Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
