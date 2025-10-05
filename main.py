import cv2
import numpy as np

# --- Seleção da câmera pelo usuário com validação
while True:
    try:
        camera_index_str = input("Digite o número da câmera que deseja usar (0, 1, 2...): ")
        camera_index = int(camera_index_str)
        break
    except ValueError:
        print("Entrada inválida. Por favor, digite um número.")

# --- Inicialização da câmera escolhida
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Não foi possível abrir a câmera {camera_index}.")
    exit()

# Obtém a resolução da câmera
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Resolução da câmera: {frame_width}x{frame_height}")

# --- Dicionário de marcadores ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# --- Parâmetros da câmera (ajustados para webcam típica)
# IMPORTANTE: Calibre sua câmera para obter valores precisos!
camera_matrix = np.array([[frame_width, 0, frame_width/2],
                          [0, frame_width, frame_height/2],
                          [0, 0, 1]], dtype=float)
dist_coeffs = np.zeros((4, 1))  # Assumindo nenhuma distorção

# --- Tamanho REAL do marcador em metros (AJUSTE ESTE VALOR!)
# Meça seu marcador impresso com uma régua! Ex: 5cm = 0.05m
MARKER_SIZE_REAL_M = 0.05

print(f"\n⚠️  TAMANHO DO MARCADOR DEFINIDO: {MARKER_SIZE_REAL_M}m ({MARKER_SIZE_REAL_M*100}cm)")
print("⚠️  Certifique-se de que este valor corresponde ao tamanho real do seu marcador impresso!")
print("\nPressione ESC para sair\n")

# --- Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # CORREÇÃO 1: aruco_dict ao invés de aruco*dict
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Estima a pose de cada marcador detectado
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE_REAL_M, camera_matrix, dist_coeffs
        )
        
        # Itera sobre todas as poses detectadas
        for i in range(len(ids)):
            rvec = rvecs[i]
            tvec = tvecs[i]
            
            # Desenha os eixos do marcador
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE_REAL_M * 0.5)
            
            # Pontos do cubo 3D no sistema de coordenadas do marcador
            half_marker = MARKER_SIZE_REAL_M / 2
            
            # Cubo desenhado ACIMA do marcador (Z positivo)
            cube_points = np.float32([
                # Base (no plano do marcador)
                [-half_marker, -half_marker, 0],
                [ half_marker, -half_marker, 0],
                [ half_marker,  half_marker, 0],
                [-half_marker,  half_marker, 0],
                # Topo (acima do marcador)
                [-half_marker, -half_marker, MARKER_SIZE_REAL_M],
                [ half_marker, -half_marker, MARKER_SIZE_REAL_M],
                [ half_marker,  half_marker, MARKER_SIZE_REAL_M],
                [-half_marker,  half_marker, MARKER_SIZE_REAL_M]
            ])
            
            # CORREÇÃO 2: cube_points ao invés de cube*points
            imgpts, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            
            # Desenha as linhas do cubo
            # Base inferior (no marcador - em verde)
            cv2.drawContours(frame, [imgpts[:4]], -1, (0, 255, 0), 3)
            
            # Pilares verticais (em azul)
            for j in range(4):
                cv2.line(frame, tuple(imgpts[j]), tuple(imgpts[j+4]), (255, 0, 0), 3)
            
            # Base superior (topo do cubo - em vermelho)
            cv2.drawContours(frame, [imgpts[4:]], -1, (0, 0, 255), 3)
            
            # Adiciona informação de distância na tela
            distance = np.linalg.norm(tvec)
            text = f"ID:{ids[i][0]} Dist:{distance:.2f}m"
            cv2.putText(frame, text, tuple(imgpts[0]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("ArUco - Cubo 3D", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Pressione ESC para sair
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Programa finalizado")