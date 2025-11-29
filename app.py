import cv2
import mediapipe as mp
import argparse
import time
from utils import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, AlarmPlayer

mp_face_mesh = mp.solutions.face_mesh

def extract_eye_points(landmarks, eye_indices, image_w, image_h):
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        x = int(lm.x * image_w)
        y = int(lm.y * image_h)
        points.append((x, y))
    return points

def main(args):
    cap = cv2.VideoCapture(0)

    alarm = AlarmPlayer(alarm_wav_path=args.alarm)
    ear_thresh = args.ear_thresh
    consec_frames = args.consec_frames
    counter = 0

    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            h, w = frame.shape[:2]

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                left_eye = extract_eye_points(lm, LEFT_EYE, w, h)
                right_eye = extract_eye_points(lm, RIGHT_EYE, w, h)

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                if ear < ear_thresh:
                    counter += 1
                else:
                    counter = 0

                if counter >= consec_frames:
                    alarm.start()
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    alarm.stop()

                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ear-thresh", type=float, default=0.23)
    parser.add_argument("--consec-frames", type=int, default=20)
    parser.add_argument("--alarm", type=str, default=None)
    args = parser.parse_args()
    main(args)
