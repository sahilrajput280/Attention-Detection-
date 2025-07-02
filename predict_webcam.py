import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

CLASSES = ['attentive', 'not_attentive', 'using_phone', 'sleeping']
CLASS_COLORS = {
    'attentive': (0, 255, 0),
    'not_attentive': (0, 0, 255),
    'using_phone': (0, 165, 255),
    'sleeping': (255, 0, 0),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = models.resnet50(pretrained=False)
classifier.fc = nn.Linear(classifier.fc.in_features, len(CLASSES))
classifier.load_state_dict(torch.load("best_resnet50.pth", map_location=device))
classifier.to(device)
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])


detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
phone_class_id = 67 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

sleep_counter = 0
SLEEP_THRESHOLD = 15

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = detector(frame)
    phone_detected = False
    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == phone_class_id and conf > 0.5:
            phone_detected = True
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), CLASS_COLORS['using_phone'], 2)
            cv2.putText(frame, f'Phone {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, CLASS_COLORS['using_phone'], 2)

    face_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    eyes_detected = 0

    for (x, y, w, h) in face_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        eyes_detected += len(eyes)   
        break

    if eyes_detected < 2:
        sleep_counter += 1
    else:
        sleep_counter = 0

    if phone_detected:
        label = 'using_phone'
        confidence = 1.0
    elif sleep_counter >= SLEEP_THRESHOLD:
        label = 'sleeping'
        confidence = 1.0
    else:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = classifier(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
            label = CLASSES[pred.item()]
            confidence = conf.item()
    
    
    color = CLASS_COLORS.get(label, (255, 255, 255))
    text = f'{label} ({confidence*100:.1f}%)'
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Attention Detection + Phone & Eye Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()