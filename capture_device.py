import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()

pre_editado_path = 'abanar.mp4'
capture_video = cv2.VideoCapture(pre_editado_path)

while True:
    ret_preditado, frame_preditado = capture_video.read()

    if not ret_preditado:
        break

    image_rgb_preditado = cv2.cvtColor(frame_preditado, cv2.COLOR_BGR2RGB)
    result_preditado = hands.process(image_rgb_preditado)

    if result_preditado.multi_hand_landmarks:
        for hand_landmarks in result_preditado.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame_preditado, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Vídeo Pré-Editado', frame_preditado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_video.release()
cv2.destroyAllWindows()
