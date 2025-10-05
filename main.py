import cv2
import numpy as np

# --- Seleção da câmera pelo usuário
print("Digite o número da câmera que deseja usar (0, 1, 2...):")
camera_index = int(input("Câmera: "))

# --- Inicialização da câmera escolhida
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Não foi possível abrir a câmera {camera_index}.")
    exit()

# --- Dicionário de marcadores ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# --- Parâmetros fictícios de calibração da câmera
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0,   0,   1]], dtype=float)
dist_coeffs = np.zeros((4, 1))

# --- Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        print("IDs detectados:", ids)
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Desenha os cantos detectados para debug
        for corner in corners:
            corner = corner.reshape((4, 2))
            for x, y in corner:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        # --- Calcula marker_size baseado na largura do marcador detectado em pixels
        # Pega o primeiro marcador detectado
        c = corners[0].reshape(4, 2)
        pixel_width = np.linalg.norm(c[0] - c[1])
        marker_size = pixel_width / 100  # escala arbitrária para projeção do cubo

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs
        )

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

            # Cubo 3D
            cube_points = np.float32([
                [0, 0, 0],
                [0, marker_size, 0],
                [marker_size, marker_size, 0],
                [marker_size, 0, 0],
                [0, 0, -marker_size],
                [0, marker_size, -marker_size],
                [marker_size, marker_size, -marker_size],
                [marker_size, 0, -marker_size]
            ])
            imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # Desenha cubo
            frame = cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 2)
            for i, j in zip(range(4), range(4, 8)):
                frame = cv2.line(frame, tuple(imgpts[i]), tuple(imgpts[j]), (255, 0, 0), 2)
            frame = cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 2)

    cv2.imshow("Debug ArUco - Cubo 3D", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
