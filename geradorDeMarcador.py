import cv2

# Escolhe o dicionário
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Gera o marcador com ID=0 e tamanho 400x400 pixels
marker_image = cv2.aruco.generateImageMarker(aruco_dict, 0, 400)

# Salva o marcador
cv2.imwrite("marcador0.png", marker_image)

print("✅ Marcador salvo como marcador0.png")
