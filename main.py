from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
import math as m

app = FastAPI()
# 🔹 Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois troque para o domínio do Lovable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------- Funções auxiliares -------------------
def convert_seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    return int(180 / m.pi) * theta

# ------------------- Endpoint principal -------------------
@app.post("/analisar")
async def analisar(video: UploadFile = File(...)):
    try:
        # Salvar vídeo temporário
        input_path = OUTPUT_DIR / video.filename
        with open(input_path, "wb") as f:
            f.write(await video.read())

        cap = cv2.VideoCapture(str(input_path))
        default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        output_video_path = OUTPUT_DIR / "video_marked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (default_width, default_height))

        mp_holistic = mp.solutions.holistic
        data = []

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                height, width, _ = image.shape
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    # Pontos principais
                    left_shoulder = [int(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * width),
                                     int(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * height)]
                    right_shoulder = [int(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * width),
                                      int(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * height)]
                    left_hip = [int(landmarks[mp_holistic.PoseLandmark.LEFT_HIP].x * width),
                                int(landmarks[mp_holistic.PoseLandmark.LEFT_HIP].y * height)]
                    right_hip = [int(landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].x * width),
                                 int(landmarks[mp_holistic.PoseLandmark.RIGHT_HIP].y * height)]
                    left_ear = [int(landmarks[mp_holistic.PoseLandmark.LEFT_EAR].x * width),
                                int(landmarks[mp_holistic.PoseLandmark.LEFT_EAR].y * height)]
                    right_ear = [int(landmarks[mp_holistic.PoseLandmark.RIGHT_EAR].x * width),
                                 int(landmarks[mp_holistic.PoseLandmark.RIGHT_EAR].y * height)]
                    elbow_left = [int(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].x * width),
                                  int(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].y * height)]
                    wrist_left = [int(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].x * width),
                                  int(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].y * height)]
                    elbow_right = [int(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * width),
                                   int(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * height)]
                    wrist_right = [int(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].x * width),
                                   int(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].y * height)]

                    left_visibility = landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].visibility
                    right_visibility = landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].visibility

                    neck_base = ((left_shoulder[0] + right_shoulder[0]) // 2,
                                 (left_shoulder[1] + right_shoulder[1]) // 2)

                    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    current_time_seconds = frame_number / fps
                    current_time_hhmmss = convert_seconds_to_hhmmss(current_time_seconds)

                    if left_visibility > right_visibility:
                        # ângulos
                        angle_tronco = calculate_angle(left_shoulder, left_hip, [left_hip[0], left_hip[1]-1])
                        angle_pescoco = findAngle(neck_base[0], neck_base[1], left_ear[0], left_ear[1])
                        angle_antebraco = calculate_angle(left_shoulder, elbow_left, wrist_left)
                        angle_braco = calculate_angle(left_hip, left_shoulder, wrist_left)

                        # textos
                        cv2.putText(image, f'Tronco: {int(angle_tronco)}°', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(image, f'Pescoco: {int(angle_pescoco)}°', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(image, f'Antebraco: {int(angle_antebraco)}°', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(image, f'Braco: {int(angle_braco)}°', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                        # pontos/linhas
                        cv2.circle(image, tuple(left_shoulder), 5, (0,255,0), -1)
                        cv2.circle(image, tuple(left_hip), 5, (0,255,0), -1)
                        cv2.circle(image, tuple(elbow_left), 5, (0,255,0), -1)
                        cv2.circle(image, tuple(wrist_left), 5, (0,255,0), -1)
                        cv2.line(image, tuple(left_shoulder), tuple(left_hip), (0,255,0), 2)
                        cv2.line(image, tuple(left_shoulder), tuple(elbow_left), (0,255,0), 2)
                        cv2.line(image, tuple(elbow_left), tuple(wrist_left), (0,255,0), 2)

                        data.append({
                            'Tempo (s)': current_time_hhmmss,
                            'Ângulo Tronco Esquerdo': angle_tronco, 'Ângulo Tronco Direito': 0,
                            'Ângulo Pescoco Esquerdo': angle_pescoco, 'Ângulo Pescoco Direito': 0,
                            'Ângulo AnteBraco Esquerdo': angle_antebraco, 'Ângulo AnteBraco Direito': 0,
                            'Ângulo Braco Esquerdo': angle_braco, 'Ângulo Braco Direito': 0,
                        })

                    else:
                        angle_tronco = calculate_angle(right_shoulder, right_hip, [right_hip[0], right_hip[1]-1])
                        angle_pescoco = findAngle(neck_base[0], neck_base[1], right_ear[0], right_ear[1])
                        angle_antebraco = calculate_angle(right_shoulder, elbow_right, wrist_right)
                        angle_braco = calculate_angle(right_hip, right_shoulder, wrist_right)

                        cv2.putText(image, f'Tronco: {int(angle_tronco)}°', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(image, f'Pescoco: {int(angle_pescoco)}°', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(image, f'Antebraco: {int(angle_antebraco)}°', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(image, f'Braco: {int(angle_braco)}°', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                        cv2.circle(image, tuple(right_shoulder), 5, (0,255,0), -1)
                        cv2.circle(image, tuple(right_hip), 5, (0,255,0), -1)
                        cv2.circle(image, tuple(elbow_right), 5, (0,255,0), -1)
                        cv2.circle(image, tuple(wrist_right), 5, (0,255,0), -1)
                        cv2.line(image, tuple(right_shoulder), tuple(right_hip), (0,255,0), 2)
                        cv2.line(image, tuple(right_shoulder), tuple(elbow_right), (0,255,0), 2)
                        cv2.line(image, tuple(elbow_right), tuple(wrist_right), (0,255,0), 2)

                        data.append({
                            'Tempo (s)': current_time_hhmmss,
                            'Ângulo Tronco Esquerdo': 0, 'Ângulo Tronco Direito': angle_tronco,
                            'Ângulo Pescoco Esquerdo': 0, 'Ângulo Pescoco Direito': angle_pescoco,
                            'Ângulo AnteBraco Esquerdo': 0, 'Ângulo AnteBraco Direito': angle_antebraco,
                            'Ângulo Braco Esquerdo': 0, 'Ângulo Braco Direito': angle_braco,
                        })

                out.write(image)

        cap.release()
        out.release()

        # Relatórios
        df = pd.DataFrame(data)
        report_path_raw = OUTPUT_DIR / "relatorio_angulo.xlsx"
        report_path_median = OUTPUT_DIR / "relatorio_angulo_median.xlsx"
        df.to_excel(report_path_raw, index=False)
        df.groupby('Tempo (s)').median().reset_index().to_excel(report_path_median, index=False)

        return JSONResponse({
            "status": "ok",
            "fileName": video.filename,
            "duration": convert_seconds_to_hhmmss(cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps),
            "videoUrl": str(output_video_path),
            "reportUrlRaw": str(report_path_raw),
            "reportUrlMedian": str(report_path_median),
            "framesProcessed": len(data)
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
