import cv2
import numpy as np
import trimesh
import pandas as pd
import os

print("=" * 60)
print("SISTEMA ARUCO + MODELO 3D + DADOS DE SATÉLITE")
print("Suporta: OBJ, STL, PLY, FBX, GLB, GLTF, 3DS, COLLADA")
print("=" * 60)

# ============================================================
# CONFIGURAÇÕES
# ============================================================
# Modelo 3D
model_path = "tartarugaReduzida.obj"

# Dados de satélite
csv_path = "dados-sateliteGiovanni.csv"

# Tamanho do marcador
MARKER_SIZE_REAL_M = 0.05

# Posição do modelo
MODEL_OFFSET_X = 0.0
MODEL_OFFSET_Y = 0.0
MODEL_OFFSET_Z = 0.05
# ============================================================

# --- Carregar dados de satélite ---
print("\n" + "=" * 60)
print("CARREGANDO DADOS DE SATÉLITE...")
print("=" * 60)

try:
    df_satellite = pd.read_csv(csv_path)
    # Renomeia coluna para facilitar
    df_satellite = df_satellite.rename(columns={'M2TMNXOCN_5_12_4_TSKINWTR': 'temperatura'})
    
    # Remove dados ausentes (pontos terrestres)
    df_satellite = df_satellite.dropna(subset=['temperatura'])
    
    print(f"✅ Dados carregados: {len(df_satellite):,} pontos oceânicos")
    print(f"   Latitude: {df_satellite['lat'].min():.2f}° a {df_satellite['lat'].max():.2f}°")
    print(f"   Longitude: {df_satellite['lon'].min():.2f}° a {df_satellite['lon'].max():.2f}°")
    print(f"   Temperatura: {df_satellite['temperatura'].min():.2f}°C a {df_satellite['temperatura'].max():.2f}°C")
    
    satellite_data_loaded = True
except FileNotFoundError:
    print(f"❌ Arquivo '{csv_path}' não encontrado.")
    print("⚠️  O programa continuará sem dados de satélite.")
    satellite_data_loaded = False
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    satellite_data_loaded = False

# --- Função para buscar dados de temperatura ---
def get_temperature_data(lat, lon):
    """
    Busca dados de temperatura para uma coordenada específica
    Retorna a temperatura do ponto mais próximo
    """
    if not satellite_data_loaded:
        return None, None, None
    
    # Calcula distância euclidiana para todos os pontos
    distances = np.sqrt((df_satellite['lat'] - lat)**2 + (df_satellite['lon'] - lon)**2)
    
    # Encontra o índice do ponto mais próximo
    nearest_idx = distances.idxmin()
    
    # Retorna dados do ponto mais próximo
    nearest_lat = df_satellite.loc[nearest_idx, 'lat']
    nearest_lon = df_satellite.loc[nearest_idx, 'lon']
    temp = df_satellite.loc[nearest_idx, 'temperatura']
    
    return nearest_lat, nearest_lon, temp

# --- Coordenadas dos marcadores (você pode mapear cada ID para uma localização) ---
# Mapeamento: ID do marcador ArUco -> (latitude, longitude)
marker_locations = {
    0: (-8.05, -34.90),   # Exemplo: Recife, PE
    1: (-23.55, -46.63),  # São Paulo
    2: (-22.91, -43.17),  # Rio de Janeiro
    3: (40.71, -74.01),   # Nova York
    4: (51.51, -0.13),    # Londres
}

# Você pode alterar para uma localização padrão se não quiser mapear por ID
USE_DEFAULT_LOCATION = True
DEFAULT_LAT = -8.05    # Recife, PE
DEFAULT_LON = -34.90

# --- Configuração da câmera ---
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

# --- ArUco ---
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

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

# --- Carregar modelo 3D ---
print("\n" + "=" * 60)
print("CARREGANDO MODELO 3D...")
print("=" * 60)

mesh_3d = None
use_model = False
model_scale = 1.0
texture_image = None

if model_path and os.path.exists(model_path):
    try:
        mesh_3d = trimesh.load(model_path)
        
        if isinstance(mesh_3d, trimesh.Scene):
            mesh_3d = trimesh.util.concatenate(
                [geom for geom in mesh_3d.geometry.values()]
            )
        
        bounds = mesh_3d.bounds
        max_dimension = np.max(bounds[1] - bounds[0])
        model_scale = MARKER_SIZE_REAL_M / max_dimension
        
        mesh_3d.apply_translation(-mesh_3d.centroid)
        
        num_faces = len(mesh_3d.faces)
        if num_faces > 10000:
            print(f"   ⚠️  ATENÇÃO: Modelo tem {num_faces} faces!")
            print(f"   💡 Dica: Use modo Wireframe (W) para melhor performance")
        
        use_model = True
        print(f"✅ Modelo carregado: {model_path}")
        print(f"   Vértices: {len(mesh_3d.vertices)}")
        print(f"   Faces: {len(mesh_3d.faces)}")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        use_model = False
else:
    if model_path:
        print(f"❌ Arquivo não encontrado: {model_path}")
    print("⚠️  Usando cubo padrão")

if not use_model:
    mesh_3d = trimesh.creation.box(extents=[MARKER_SIZE_REAL_M] * 3)
    model_scale = 1.0

# --- Funções de renderização ---
def draw_mesh_on_frame(frame, mesh, rvec, tvec, camera_matrix, dist_coeffs, scale=1.0, offset=(0, 0, 0)):
    vertices_3d = mesh.vertices * scale
    vertices_3d = vertices_3d + np.array(offset)
    
    R, _ = cv2.Rodrigues(rvec)
    vertices_cam = np.dot(vertices_3d, R.T) + tvec
    
    vertices_2d, _ = cv2.projectPoints(
        vertices_cam, np.zeros((3,1)), np.zeros((3,1)), 
        camera_matrix, dist_coeffs
    )
    vertices_2d = vertices_2d.reshape(-1, 2).astype(int)
    
    vertex_colors = None
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        vertex_colors = mesh.visual.vertex_colors
    
    num_faces = len(mesh.faces)
    skip_factor = 1
    if num_faces > 20000:
        skip_factor = 4
    elif num_faces > 10000:
        skip_factor = 3
    elif num_faces > 5000:
        skip_factor = 2
    
    for face_idx, face in enumerate(mesh.faces):
        if skip_factor > 1 and face_idx % skip_factor != 0:
            continue
            
        pts = vertices_2d[face]
        face_vertices = vertices_cam[face]
        
        if not np.all(face_vertices[:, 2] > 0):
            continue
        
        if (np.any(pts[:, 0] < -100) or np.any(pts[:, 0] > frame.shape[1] + 100) or
            np.any(pts[:, 1] < -100) or np.any(pts[:, 1] > frame.shape[0] + 100)):
            continue
        
        if vertex_colors is not None:
            face_colors = vertex_colors[face]
            avg_color = np.mean(face_colors[:, :3], axis=0).astype(int)
            color = tuple(map(int, avg_color[::-1]))
        else:
            color = (100, 150, 200)
        
        cv2.fillConvexPoly(frame, pts, color)
        
        if num_faces < 2000:
            cv2.polylines(frame, [pts], True, (50, 50, 50), 1)
    
    return frame

def draw_wireframe(frame, mesh, rvec, tvec, camera_matrix, dist_coeffs, scale=1.0, offset=(0, 0, 0)):
    vertices_3d = mesh.vertices * scale
    vertices_3d = vertices_3d + np.array(offset)
    
    R, _ = cv2.Rodrigues(rvec)
    vertices_cam = np.dot(vertices_3d, R.T) + tvec
    
    vertices_2d, _ = cv2.projectPoints(
        vertices_cam, np.zeros((3,1)), np.zeros((3,1)), 
        camera_matrix, dist_coeffs
    )
    vertices_2d = vertices_2d.reshape(-1, 2).astype(int)
    
    edges = mesh.edges_unique
    num_edges = len(edges)
    
    skip_factor = 1
    if num_edges > 30000:
        skip_factor = 5
    elif num_edges > 15000:
        skip_factor = 4
    elif num_edges > 8000:
        skip_factor = 3
    elif num_edges > 3000:
        skip_factor = 2
    
    for i, edge in enumerate(edges):
        if skip_factor > 1 and i % skip_factor != 0:
            continue
            
        p1, p2 = vertices_2d[edge]
        
        if vertices_cam[edge[0], 2] > 0 and vertices_cam[edge[1], 2] > 0:
            if (0 <= p1[0] < frame.shape[1] and 0 <= p1[1] < frame.shape[0] and
                0 <= p2[0] < frame.shape[1] and 0 <= p2[1] < frame.shape[0]):
                cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 0), 1)
    
    return frame

# --- Função para desenhar painel de informações ---
def draw_info_panel(frame, lat, lon, temp, marker_id, distance):
    """
    Desenha um painel com informações dos dados de satélite
    """
    # Posição do painel (canto inferior esquerdo)
    panel_x = 10
    panel_y = frame_height - 180
    panel_width = 400
    panel_height = 170
    
    # Desenha fundo do painel (semi-transparente)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height),
                 (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Borda do painel
    cv2.rectangle(frame, (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height),
                 (255, 255, 255), 2)
    
    # Título
    cv2.putText(frame, "DADOS DE SATELITE", (panel_x + 10, panel_y + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    y_offset = panel_y + 55
    line_height = 30
    
    if temp is not None:
        # Latitude
        cv2.putText(frame, f"Latitude:  {lat:.4f}", (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Longitude
        y_offset += line_height
        cv2.putText(frame, f"Longitude: {lon:.4f}", (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Temperatura com cor baseada no valor
        y_offset += line_height
        if temp < 10:
            temp_color = (255, 100, 100)  # Azul para frio
        elif temp < 20:
            temp_color = (0, 255, 255)    # Amarelo para temperado
        else:
            temp_color = (0, 165, 255)    # Laranja para quente
        
        cv2.putText(frame, f"Temperatura: {temp:.2f} C", (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, temp_color, 2)
        
        # Info adicional
        y_offset += line_height
        cv2.putText(frame, f"Marcador ID: {marker_id} | Dist: {distance:.1f}cm", 
                   (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "Dados nao disponiveis", (panel_x + 15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    return frame

print("\n" + "=" * 60)
print("CONTROLES:")
print("  ESC - Sair")
print("  W - Toggle Wireframe/Sólido")
print("  + - Aumentar escala")
print("  - - Diminuir escala")
print("  R - Resetar escala")
print("  Setas - Mover modelo (↑↓←→)")
print("  PgUp/PgDn - Altura do modelo")
print("=" * 60)

wireframe_mode = True
user_scale = 1.0

print(f"\n⚡ Modo inicial: Wireframe")

# --- Loop principal ---
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
            marker_id = ids[i][0]
            
            # Desenha eixos
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                            rvec, tvec, MARKER_SIZE_REAL_M * 0.5)
            
            # Desenha modelo 3D
            final_scale = model_scale * user_scale
            model_offset = (MODEL_OFFSET_X, MODEL_OFFSET_Y, MODEL_OFFSET_Z)
            
            if wireframe_mode:
                frame = draw_wireframe(frame, mesh_3d, rvec, tvec, 
                                     camera_matrix, dist_coeffs, final_scale, model_offset)
            else:
                frame = draw_mesh_on_frame(frame, mesh_3d, rvec, tvec, 
                                          camera_matrix, dist_coeffs, final_scale, model_offset)
            
            # Obtém localização para este marcador
            if USE_DEFAULT_LOCATION:
                target_lat = DEFAULT_LAT
                target_lon = DEFAULT_LON
            elif marker_id in marker_locations:
                target_lat, target_lon = marker_locations[marker_id]
            else:
                target_lat, target_lon = DEFAULT_LAT, DEFAULT_LON
            
            # Busca dados de temperatura
            nearest_lat, nearest_lon, temp = get_temperature_data(target_lat, target_lon)
            
            # Calcula distância do marcador
            distance = np.linalg.norm(tvec) * 100  # em cm
            
            # Desenha painel de informações
            if nearest_lat is not None:
                frame = draw_info_panel(frame, nearest_lat, nearest_lon, temp, marker_id, distance)
            
            # Info na tela superior
            info_text = f"Marcador ID:{marker_id} | Escala:{user_scale:.2f}x"
            cv2.putText(frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Nenhum marcador detectado", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Modo de renderização
    mode_text = "Wireframe" if wireframe_mode else "Solido"
    cv2.putText(frame, f"Modo: {mode_text}", (10, frame_height - 200),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("ArUco + Modelo 3D + Dados Satelite", frame)
    
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
    elif key == 82:  # Seta para cima
        MODEL_OFFSET_Y -= 0.005
        print(f"Offset Y: {MODEL_OFFSET_Y:.3f}m")
    elif key == 84:  # Seta para baixo
        MODEL_OFFSET_Y += 0.005
        print(f"Offset Y: {MODEL_OFFSET_Y:.3f}m")
    elif key == 81:  # Seta esquerda
        MODEL_OFFSET_X -= 0.005
        print(f"Offset X: {MODEL_OFFSET_X:.3f}m")
    elif key == 83:  # Seta direita
        MODEL_OFFSET_X += 0.005
        print(f"Offset X: {MODEL_OFFSET_X:.3f}m")
    elif key == 85:  # Page Up
        MODEL_OFFSET_Z += 0.005
        print(f"Altura Z: {MODEL_OFFSET_Z:.3f}m")
    elif key == 86:  # Page Down
        MODEL_OFFSET_Z -= 0.005
        print(f"Altura Z: {MODEL_OFFSET_Z:.3f}m")

cap.release()
cv2.destroyAllWindows()
print("\n✅ Programa finalizado")