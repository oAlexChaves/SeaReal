import cv2
import numpy as np
import trimesh
import os

print("=" * 60)
print("SISTEMA ARUCO + MODELO 3D (VERSÃO SIMPLES)")
print("Suporta: OBJ, STL, PLY, FBX, GLB, GLTF, 3DS, COLLADA")
print("=" * 60)

# --- Configuração da câmera
while True:
    try:
        camera_index_str = input("\nDigite o número da câmera (0, 1, 2...): ")
        camera_index = int(camera_index_str)
        break
    except ValueError:
        print("❌ Entrada inválida. Digite um número.")

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"❌ Não foi possível abrir a câmera {camera_index}.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"\n✅ Câmera: {frame_width}x{frame_height}")

# --- ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Parâmetros relaxados
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# Matriz da câmera
focal_length = max(frame_width, frame_height)
camera_matrix = np.array([
    [focal_length, 0, frame_width/2],
    [0, focal_length, frame_height/2],
    [0, 0, 1]
], dtype=float)
dist_coeffs = np.zeros((5, 1))

MARKER_SIZE_REAL_M = 0.05

# --- Carregar modelo 3D
print("\n" + "=" * 60)
print("CARREGANDO MODELO 3D...")
print("=" * 60)

# ============================================================
# CONFIGURE O CAMINHO DO SEU MODELO AQUI:
# ============================================================
model_path = "tartaruga2.obj"  # <-- COLOQUE O CAMINHO DO SEU MODELO AQUI
# Exemplos:
# model_path = "C:\\Users\\SeuNome\\Desktop\\meu_modelo.obj"
# model_path = "/home/user/Downloads/carro.glb"
# model_path = "models/personagem.stl"
# model_path = ""  # Deixe vazio para usar cubo padrão
# ============================================================

print(f"Caminho configurado: {model_path if model_path else 'Cubo padrão'}")

mesh_3d = None
use_model = False
model_scale = 1.0

if model_path and os.path.exists(model_path):
    try:
        # Trimesh carrega automaticamente vários formatos
        mesh_3d = trimesh.load(model_path)
        
        # Se for uma cena (múltiplos objetos), pega a geometria
        if isinstance(mesh_3d, trimesh.Scene):
            mesh_3d = trimesh.util.concatenate(
                [geom for geom in mesh_3d.geometry.values()]
            )
        
        # Normaliza o modelo para caber no marcador
        bounds = mesh_3d.bounds
        max_dimension = np.max(bounds[1] - bounds[0])
        model_scale = MARKER_SIZE_REAL_M / max_dimension
        
        # Centraliza o modelo
        mesh_3d.apply_translation(-mesh_3d.centroid)
        
        use_model = True
        print(f"✅ Modelo carregado: {model_path}")
        print(f"   Vértices: {len(mesh_3d.vertices)}")
        print(f"   Faces: {len(mesh_3d.faces)}")
        print(f"   Escala automática: {model_scale:.4f}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        print("⚠️  Usando cubo padrão")
        use_model = False
else:
    if model_path:
        print(f"❌ Arquivo não encontrado: {model_path}")
    print("⚠️  Usando cubo padrão")

# Se não carregou modelo, cria um cubo simples
if not use_model:
    mesh_3d = trimesh.creation.box(extents=[MARKER_SIZE_REAL_M] * 3)
    model_scale = 1.0

# --- Função para projetar e desenhar malha 3D
def draw_mesh_on_frame(frame, mesh, rvec, tvec, camera_matrix, dist_coeffs, scale=1.0):
    """
    Desenha a malha 3D no frame usando projeção 2D
    """
    # Aplica escala
    vertices_3d = mesh.vertices * scale
    
    # Converte rodrigues para matriz de rotação
    R, _ = cv2.Rodrigues(rvec)
    
    # Transforma vértices para o sistema de coordenadas da câmera
    vertices_cam = np.dot(vertices_3d, R.T) + tvec
    
    # Projeta para 2D
    vertices_2d, _ = cv2.projectPoints(
        vertices_cam, 
        np.zeros((3,1)), 
        np.zeros((3,1)), 
        camera_matrix, 
        dist_coeffs
    )
    vertices_2d = vertices_2d.reshape(-1, 2).astype(int)
    
    # Desenha as arestas da malha
    for face in mesh.faces:
        pts = vertices_2d[face]
        
        # Verifica se está na frente da câmera
        face_vertices = vertices_cam[face]
        if np.all(face_vertices[:, 2] > 0):
            # Desenha triângulo preenchido com cor
            color = (0, 150, 255)
            cv2.fillPoly(frame, [pts], color)
            
            # Desenha contorno
            cv2.polylines(frame, [pts], True, (0, 0, 0), 1)
    
    return frame

# --- Função alternativa: wireframe simples
def draw_wireframe(frame, mesh, rvec, tvec, camera_matrix, dist_coeffs, scale=1.0):
    """
    Desenha apenas as arestas (wireframe)
    """
    vertices_3d = mesh.vertices * scale
    R, _ = cv2.Rodrigues(rvec)
    vertices_cam = np.dot(vertices_3d, R.T) + tvec
    
    vertices_2d, _ = cv2.projectPoints(
        vertices_cam, 
        np.zeros((3,1)), 
        np.zeros((3,1)), 
        camera_matrix, 
        dist_coeffs
    )
    vertices_2d = vertices_2d.reshape(-1, 2).astype(int)
    
    # Desenha arestas únicas
    edges = mesh.edges_unique
    for edge in edges:
        p1, p2 = vertices_2d[edge]
        # Verifica se está na frente da câmera
        if vertices_cam[edge[0], 2] > 0 and vertices_cam[edge[1], 2] > 0:
            cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 0), 2)
    
    return frame

print("\n" + "=" * 60)
print("CONTROLES:")
print("  ESC - Sair")
print("  W - Toggle Wireframe/Sólido")
print("  + - Aumentar escala")
print("  - - Diminuir escala")
print("  R - Resetar escala")
print("=" * 60)

wireframe_mode = False
user_scale = 1.0

# --- Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE_REAL_M, camera_matrix, dist_coeffs
        )
        
        for i in range(len(ids)):
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]
            
            # Desenha eixos
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                            rvec, tvec, MARKER_SIZE_REAL_M * 0.5)
            
            # Desenha modelo 3D
            final_scale = model_scale * user_scale
            
            if wireframe_mode:
                frame = draw_wireframe(frame, mesh_3d, rvec, tvec, 
                                     camera_matrix, dist_coeffs, final_scale)
            else:
                frame = draw_mesh_on_frame(frame, mesh_3d, rvec, tvec, 
                                          camera_matrix, dist_coeffs, final_scale)
            
            # Info na tela
            distance = np.linalg.norm(tvec)
            info_text = f"ID:{ids[i][0]} | Dist:{distance*100:.1f}cm | Escala:{user_scale:.2f}x"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Nenhum marcador detectado", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Modo de renderização
    mode_text = "Wireframe" if wireframe_mode else "Solido"
    cv2.putText(frame, f"Modo: {mode_text}", (10, frame_height - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("ArUco + Modelo 3D (Simples)", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == ord('w') or key == ord('W'):
        wireframe_mode = not wireframe_mode
        print(f"Modo: {'Wireframe' if wireframe_mode else 'Sólido'}")
    elif key == ord('+') or key == ord('='):
        user_scale *= 1.2
        print(f"Escala: {user_scale:.2f}x")
    elif key == ord('-') or key == ord('_'):
        user_scale /= 1.2
        print(f"Escala: {user_scale:.2f}x")
    elif key == ord('r') or key == ord('R'):
        user_scale = 1.0
        print("Escala resetada")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Programa finalizado")