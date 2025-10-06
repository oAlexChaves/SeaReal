# SeaReal

Este projeto implementa uma **sistema de Realidade Aumentada (AR)** utilizando **marcadores ArUco** e **modelos 3D**.
A aplicação detecta marcadores em tempo real através da câmera, estima sua pose e **projeta um modelo 3D sobre o marcador**.

O projeto foi desenvolvido em **Python**, com uso das bibliotecas **OpenCV**, **NumPy** e **Trimesh**.
Suporta múltiplos formatos 3D, incluindo: `.OBJ`, `.STL`, `.PLY`, `.FBX`, `.GLB`, `.GLTF`, `.3DS`, e `.DAE (Collada)`.

---

## 🚀 Funcionalidades

✅ Detecção de marcadores **ArUco** em tempo real  
✅ Estimação de **pose 3D (rotação e translação)** da câmera  
✅ Renderização de **modelos 3D reais** sobre o marcador  
✅ Alternância entre modo **wireframe** e **sólido**  
✅ Suporte a **texturas (UV mapping)**  
✅ Ajuste dinâmico de escala e posição do modelo  
✅ Compatibilidade com diversos formatos de arquivo 3D  
✅ Interface leve e interativa via teclado  

---

## 🧩 Tecnologias Utilizadas

| Biblioteca | Função |
|-------------|--------|
| `opencv-python` | Captura de vídeo, detecção de marcadores e projeção 3D |
| `numpy` | Processamento numérico e vetorial |
| `trimesh` | Leitura, manipulação e normalização de modelos 3D |
| `os` | Verificação e manipulação de caminhos de arquivos |

---

## 🧠 Conceito do Sistema

O sistema usa a biblioteca **OpenCV ArUco** para detectar marcadores visuais (QR-like).
Após a detecção, ele calcula a **pose 3D** (posição e orientação no espaço) do marcador em relação à câmera.

Com base nessa pose, o **modelo 3D** carregado é projetado sobre o marcador — permitindo **sobreposição realista** no vídeo em tempo real.

---

## 🛠️ Instalação

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/oAlexChaves/SeaReal
cd SeaReal
```

### 2️⃣ Criar e ativar o ambiente virtual
```bash
python -m venv venv
```

- **Windows (PowerShell):**
  ```bash
venv\Scripts\activate
```
- **Linux/Mac:**
  ```bash
source venv/bin/activate
```

### 3️⃣ Instalar dependências
```bash
pip install opencv-python numpy trimesh
```

> 💡 Para suportar todos os formatos 3D (como `.FBX`, `.GLTF` e `.3DS`), instale também:
```bash
pip install pyglet pycollada
```

### 4️⃣ Instalar todas as dependências do projeto (via arquivo `requirements.txt`)
Caso você tenha o arquivo `requirements.txt` exportado do seu ambiente anterior:
```bash
pip install -r requirements.txt
```

---

## 🎮 Como Usar

1. **Coloque um marcador ArUco 6x6_250** impresso ou exibido na tela.  

2. **Edite o caminho do modelo no código:**
```python
model_path = "tartarugaReduzida.obj"  # coloque o caminho do seu modelo
```

3. **Execute o script:**
```bash
python main.py
```

4. **Selecione a câmera** (0, 1, 2...) conforme o dispositivo conectado.

5. **Aponte a câmera para o marcador** e veja o modelo aparecer no mundo real 🧱✨

---

## 🎹 Controles do Teclado

| Tecla | Função |
|-------|---------|
| `ESC` | Sair do programa |
| `W` | Alternar entre modo Wireframe / Sólido |
| `T` | Alternar textura ON/OFF |
| `+` / `-` | Aumentar / diminuir escala |
| `R` | Resetar escala |
| `↑` / `↓` | Mover modelo para frente / trás |
| `←` / `→` | Mover modelo para esquerda / direita |
| `PgUp` / `PgDn` | Ajustar altura do modelo (Z) |

---

## 🖼️ Exemplo de Configuração

```python
# Caminho do modelo
model_path = "tartarugaReduzida.obj"

# Deslocamento (em metros)
MODEL_OFFSET_X = 0.0
MODEL_OFFSET_Y = 0.0
MODEL_OFFSET_Z = 0.05
```

---

## ⚙️ Estrutura do Projeto

```
📂 aruco-3d-viewer
 ├── main.py                  # Código principal do sistema
 ├── tartarugaReduzida.obj    # Modelo 3D
 ├── marcador_com_margem.png  # Marcador usado pela câmera
 ├── requirements.txt         # Lista de dependências do projeto
 ├── README.md                # Este arquivo
```

---

## 🧪 Testado em

- **Python:** 3.12  
- **OpenCV:** 4.10+  
- **Sistemas:** Windows 10  
- **Câmeras:** Webcam integrada / USB  

---

## ⚡ Possíveis Melhorias

🔹 Renderização via **OpenGL/PyOpenGL** para sombras e texturas realistas  
🔹 Suporte a **múltiplos marcadores** simultâneos  
🔹 Filtro de suavização de pose (Kalman / EMA)  
🔹 Interface GUI para seleção de modelos  
🔹 Suporte a animações de modelos GLTF  

---

