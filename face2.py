import cv2
import numpy as np
import os
import time
import pickle
import math


# ══════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
DATA_FILE   = "personas.pkl"
PHOTOS_DIR  = "fotos_registradas"
UMBRAL      = 0.52
FOTOS_REG   = 6          # fotos que toma al registrar
WIN_W       = 1100       # ancho total de la ventana
WIN_H       = 680        # alto total de la ventana
CAM_W       = 760        # ancho del área de cámara
CAM_H       = 540        # alto del área de cámara

# Paleta de colores (BGR)
C_BG        = (18,  18,  24 )
C_PANEL     = (28,  28,  38 )
C_ACCENT    = (0,   200, 140)   # verde menta
C_ACCENT2   = (220, 140, 0  )   # ámbar
C_DANGER    = (60,  80,  220)   # rojo-azul
C_TEXT      = (230, 230, 235)
C_SUBTEXT   = (130, 130, 145)
C_BORDER    = (50,  50,  65 )
C_KNOWN     = (0,   210, 130)
C_UNKNOWN   = (60,  80,  220)
C_INPUT_BG  = (38,  38,  52 )
C_INPUT_ACT = (48,  48,  65 )
BLANCO      = (255, 255, 255)

F_TITLE  = cv2.FONT_HERSHEY_DUPLEX
F_BODY   = cv2.FONT_HERSHEY_SIMPLEX
F_MONO   = cv2.FONT_HERSHEY_PLAIN


# ══════════════════════════════════════════════════════════════
#  ESTADO GLOBAL DE LA INTERFAZ
# ══════════════════════════════════════════════════════════════
class Estado:
    def __init__(self):
        self.modo         = "normal"      # normal | registro | confirmacion
        self.nombre_input = ""            # texto que escribe el usuario
        self.input_activo = False
        self.cursor_blink = 0.0
        self.msg          = ""            # mensaje temporal en pantalla
        self.msg_tiempo   = 0.0
        self.msg_color    = C_ACCENT
        self.progreso_reg = 0             # cuántas fotos se tomaron
        self.animacion    = 0.0           # para efectos visuales
        self.personas_reg = []            # lista de personas para el panel
        self.hover_btn    = None          # botón bajo el mouse
        self.ultimo_click = 0.0

estado = Estado()


# ══════════════════════════════════════════════════════════════
#  DETECTOR DE CARAS
# ══════════════════════════════════════════════════════════════
def cargar_detector():
    paths = [
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ]
    for p in paths:
        if os.path.exists(p):
            d = cv2.CascadeClassifier(p)
            if not d.empty():
                return d
    raise RuntimeError("No se encontró haarcascade. Reinstala opencv-python.")


def detectar_caras(gray, detector):
    caras = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(55, 55)
    )
    return caras if len(caras) > 0 else []


# ══════════════════════════════════════════════════════════════
#  DESCRIPTOR FACIAL
# ══════════════════════════════════════════════════════════════
def extraer_descriptor(face_gray):
    img = cv2.resize(face_gray, (64, 64)).astype(np.float32) / 255.0
    features = []

    # LBP
    lbp = np.zeros(256, dtype=np.float32)
    for r in range(1, 63):
        for c in range(1, 63):
            centro = img[r, c]
            code = 0
            for i, (dr, dc) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
                if img[r+dr, c+dc] >= centro:
                    code |= (1 << i)
            lbp[code] += 1
    lbp /= (lbp.sum() + 1e-7)
    features.extend(lbp[:64])

    # HOG
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) * 180 / np.pi % 180
    for r in range(0, 64, 16):
        for c in range(0, 64, 16):
            h, _ = np.histogram(ang[r:r+16, c:c+16], bins=9,
                                 range=(0,180), weights=mag[r:r+16, c:c+16])
            h /= (h.sum() + 1e-7)
            features.extend(h)

    # Estadísticas
    for r0, r1, c0, c1 in [(0,32,0,32),(0,32,32,64),(32,64,0,32),(32,64,32,64),(16,48,16,48)]:
        z = img[r0:r1, c0:c1]
        features.extend([z.mean(), z.std()])

    feat = np.array(features, dtype=np.float32)
    feat = feat[:256] if len(feat) >= 256 else np.pad(feat, (0, 256-len(feat)))
    norm = np.linalg.norm(feat)
    return feat / (norm + 1e-7)


def similitud(a, b):
    return float(np.dot(a, b))


# ══════════════════════════════════════════════════════════════
#  BASE DE DATOS
# ══════════════════════════════════════════════════════════════
def cargar_personas():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def guardar_personas(personas):
    with open(DATA_FILE, "wb") as f:
        pickle.dump(personas, f)

def identificar(descriptor, personas):
    mejor, score = "Desconocido", 0.0
    for nombre, descs in personas.items():
        s = max(similitud(descriptor, d) for d in descs)
        if s > score:
            score, mejor = s, nombre
    if score < UMBRAL:
        return "Desconocido", 0.0
    confianza = min(100.0, (score - UMBRAL) / (1.0 - UMBRAL) * 100)
    return mejor, confianza


# ══════════════════════════════════════════════════════════════
#  PRIMITIVAS GRÁFICAS
# ══════════════════════════════════════════════════════════════
def rect_redondeado(img, x1, y1, x2, y2, r, color, grosor=-1, alpha=1.0):
    """Dibuja un rectángulo con esquinas redondeadas."""
    if alpha < 1.0:
        overlay = img.copy()
        _rect_redondeado_directo(overlay, x1, y1, x2, y2, r, color, grosor)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        _rect_redondeado_directo(img, x1, y1, x2, y2, r, color, grosor)

def _rect_redondeado_directo(img, x1, y1, x2, y2, r, color, grosor):
    r = min(r, (x2-x1)//2, (y2-y1)//2)
    if grosor == -1:
        cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
        cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
        for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img, (cx,cy), r, color, -1)
    else:
        cv2.line(img, (x1+r,y1), (x2-r,y1), color, grosor)
        cv2.line(img, (x1+r,y2), (x2-r,y2), color, grosor)
        cv2.line(img, (x1,y1+r), (x1,y2-r), color, grosor)
        cv2.line(img, (x2,y1+r), (x2,y2-r), color, grosor)
        cv2.ellipse(img, (x1+r,y1+r), (r,r), 180, 0, 90, color, grosor)
        cv2.ellipse(img, (x2-r,y1+r), (r,r), 270, 0, 90, color, grosor)
        cv2.ellipse(img, (x1+r,y2-r), (r,r),  90, 0, 90, color, grosor)
        cv2.ellipse(img, (x2-r,y2-r), (r,r),   0, 0, 90, color, grosor)

def texto_centrado(img, texto, cx, cy, font, scale, color, grosor=1):
    (tw, th), _ = cv2.getTextSize(texto, font, scale, grosor)
    cv2.putText(img, texto, (cx - tw//2, cy + th//2), font, scale, color, grosor, cv2.LINE_AA)

def barra_progreso(img, x, y, w, h, progreso, color_fondo, color_barra, r=6):
    rect_redondeado(img, x, y, x+w, y+h, r, color_fondo)
    if progreso > 0:
        bw = max(r*2, int(w * progreso))
        rect_redondeado(img, x, y, x+bw, y+h, r, color_barra)


# ══════════════════════════════════════════════════════════════
#  BOTONES
# ══════════════════════════════════════════════════════════════
BOTONES = {
    "registrar": {"label": "＋  Registrar", "x": WIN_W-310, "y": 560, "w": 280, "h": 48},
    "eliminar":  {"label": "✕  Eliminar",   "x": WIN_W-310, "y": 618, "w": 280, "h": 48},
}
BTN_REG_OK    = {"label": "✓  Guardar",    "x": WIN_W-310, "y": 490, "w": 280, "h": 48}
BTN_REG_CANCEL= {"label": "✕  Cancelar",   "x": WIN_W-310, "y": 548, "w": 280, "h": 48}

def dibujar_boton(canvas, btn, hover=False, activo=False, color=None):
    x, y, w, h = btn["x"], btn["y"], btn["w"], btn["h"]
    bg = color if color else (C_ACCENT if not hover else tuple(min(255, c+30) for c in C_ACCENT))
    if activo:
        bg = C_ACCENT2
    rect_redondeado(canvas, x, y, x+w, y+h, 10, bg)
    texto_centrado(canvas, btn["label"], x+w//2, y+h//2, F_BODY, 0.52, BLANCO, 1)

def punto_en_boton(mx, my, btn):
    x, y, w, h = btn["x"], btn["y"], btn["w"], btn["h"]
    return x <= mx <= x+w and y <= my <= y+h


# ══════════════════════════════════════════════════════════════
#  PANEL LATERAL DERECHO
# ══════════════════════════════════════════════════════════════
def dibujar_panel(canvas, personas, estado):
    px = CAM_W + 20
    pw = WIN_W - px - 10
    t  = time.time()

    # Fondo del panel
    rect_redondeado(canvas, px-5, 10, WIN_W-5, WIN_H-10, 14, C_PANEL)

    # Título
    cv2.putText(canvas, "FACE ID", (px+10, 48), F_TITLE, 0.9, C_ACCENT, 1, cv2.LINE_AA)
    cv2.line(canvas, (px+10, 58), (WIN_W-20, 58), C_BORDER, 1)

    # ── Modo NORMAL ───────────────────────────────────────────
    if estado.modo == "normal":

        # Lista de personas registradas
        cv2.putText(canvas, "Personas registradas", (px+10, 85),
                    F_BODY, 0.42, C_SUBTEXT, 1, cv2.LINE_AA)

        if not personas:
            cv2.putText(canvas, "Ninguna todavía", (px+10, 115),
                        F_BODY, 0.45, C_SUBTEXT, 1, cv2.LINE_AA)
        else:
            for i, nombre in enumerate(list(personas.keys())[:8]):
                py = 105 + i * 38
                # Tarjeta de persona
                rect_redondeado(canvas, px+6, py-18, WIN_W-16, py+14, 8, C_BG)
                # Ícono circular
                cv2.circle(canvas, (px+26, py-2), 12, C_ACCENT, -1)
                inicial = nombre[0].upper()
                texto_centrado(canvas, inicial, px+26, py-2, F_BODY, 0.45, C_BG, 1)
                # Nombre
                cv2.putText(canvas, nombre[:16], (px+46, py+4),
                            F_BODY, 0.48, C_TEXT, 1, cv2.LINE_AA)
                n_muestras = len(personas[nombre])
                cv2.putText(canvas, f"{n_muestras} muestras", (px+46, py-10),
                            F_BODY, 0.35, C_SUBTEXT, 1, cv2.LINE_AA)

        # Botones
        hover_reg = estado.hover_btn == "registrar"
        hover_del = estado.hover_btn == "eliminar"
        dibujar_boton(canvas, BOTONES["registrar"], hover=hover_reg)
        dibujar_boton(canvas, BOTONES["eliminar"],  hover=hover_del, color=C_DANGER)

    # ── Modo REGISTRO ─────────────────────────────────────────
    elif estado.modo == "registro":

        cv2.putText(canvas, "Nueva persona", (px+10, 82),
                    F_BODY, 0.5, C_SUBTEXT, 1, cv2.LINE_AA)

        # Campo de nombre
        cv2.putText(canvas, "Nombre:", (px+10, 115),
                    F_BODY, 0.45, C_TEXT, 1, cv2.LINE_AA)

        # Input box
        ib_color = C_INPUT_ACT if estado.input_activo else C_INPUT_BG
        rect_redondeado(canvas, px+8, 122, WIN_W-18, 158, 8, ib_color)
        rect_redondeado(canvas, px+8, 122, WIN_W-18, 158, 8, C_ACCENT if estado.input_activo else C_BORDER, 1)

        # Texto del input + cursor parpadeante
        cursor = "|" if (t % 1.0 < 0.5) and estado.input_activo else " "
        display = estado.nombre_input + cursor
        cv2.putText(canvas, display, (px+18, 147),
                    F_BODY, 0.62, C_TEXT, 1, cv2.LINE_AA)

        if not estado.nombre_input:
            cv2.putText(canvas, "Escribe el nombre...", (px+18, 147),
                        F_BODY, 0.52, C_SUBTEXT, 1, cv2.LINE_AA)

        # Instrucción
        cv2.putText(canvas, "Haz clic en el cuadro y escribe", (px+10, 175),
                    F_BODY, 0.38, C_SUBTEXT, 1, cv2.LINE_AA)

        # Indicador de captura
        if estado.progreso_reg > 0:
            cv2.putText(canvas, "Capturando fotos...", (px+10, 205),
                        F_BODY, 0.45, C_ACCENT2, 1, cv2.LINE_AA)
            pct = estado.progreso_reg / FOTOS_REG
            barra_progreso(canvas, px+8, 215, pw-16, 16, pct, C_BG, C_ACCENT2, 6)
            cv2.putText(canvas, f"{estado.progreso_reg}/{FOTOS_REG}",
                        (px + pw//2 - 12, 230), F_BODY, 0.4, BLANCO, 1, cv2.LINE_AA)

            # Instrucción de poses
            tips = ["Mira de frente", "Gira un poco a la derecha",
                    "Gira a la izquierda", "Inclina la cabeza", "Sonríe", "Expresión neutral"]
            tip_idx = min(estado.progreso_reg, len(tips)-1)
            cv2.putText(canvas, f"→ {tips[tip_idx]}", (px+10, 260),
                        F_BODY, 0.42, C_ACCENT, 1, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Posicionate frente a la", (px+10, 210),
                        F_BODY, 0.42, C_SUBTEXT, 1, cv2.LINE_AA)
            cv2.putText(canvas, "camara y presiona Guardar", (px+10, 230),
                        F_BODY, 0.42, C_SUBTEXT, 1, cv2.LINE_AA)

        # Botones
        nombre_ok = len(estado.nombre_input.strip()) >= 2
        hover_ok  = estado.hover_btn == "reg_ok"
        hover_can = estado.hover_btn == "reg_cancel"
        dibujar_boton(canvas, BTN_REG_OK,     hover=hover_ok,  activo=not nombre_ok,
                      color=C_ACCENT if nombre_ok else C_BORDER)
        dibujar_boton(canvas, BTN_REG_CANCEL, hover=hover_can, color=C_DANGER)


# ══════════════════════════════════════════════════════════════
#  ÁREA DE CÁMARA
# ══════════════════════════════════════════════════════════════
def dibujar_area_camara(canvas, frame_cam, caras_info, estado):
    t = time.time()

    # Fondo
    rect_redondeado(canvas, 5, 10, CAM_W+5, CAM_H+25, 14, C_PANEL)

    if frame_cam is not None:
        # Escalar el frame al área de cámara
        cam_resized = cv2.resize(frame_cam, (CAM_W-10, CAM_H-10))
        canvas[15:CAM_H+5, 10:CAM_W] = cam_resized

    # Dibujar detecciones sobre el área de cámara
    for info in caras_info:
        x, y, w, h = info["bbox_scaled"]
        nombre     = info["nombre"]
        confianza  = info["confianza"]
        color      = C_KNOWN if nombre != "Desconocido" else C_UNKNOWN

        # Caja de la cara
        cv2.rectangle(canvas, (x+10, y+15), (x+w+10, y+h+15), color, 2)

        # Esquinas decorativas
        cl = max(w, h) // 7
        for px2, py2, dx, dy in [(x+10,y+15,1,1),(x+w+10,y+15,-1,1),
                                   (x+10,y+h+15,1,-1),(x+w+10,y+h+15,-1,-1)]:
            cv2.line(canvas, (px2, py2), (px2+dx*cl, py2), color, 3)
            cv2.line(canvas, (px2, py2), (px2, py2+dy*cl), color, 3)

        # Etiqueta flotante
        if nombre == "Desconocido":
            label = "Desconocido"
        else:
            label = f"{nombre}  {confianza:.0f}%"

        (tw, th), bl = cv2.getTextSize(label, F_BODY, 0.58, 1)
        lx = x + 10
        ly = max(y + 15, th + 20)
        rect_redondeado(canvas, lx-2, ly-th-6, lx+tw+8, ly+bl+2, 6, color, alpha=0.8)
        cv2.putText(canvas, label, (lx+3, ly), F_BODY, 0.58, BLANCO, 1, cv2.LINE_AA)

    # Modo registro: marco animado verde
    if estado.modo == "registro" and estado.progreso_reg > 0:
        pulse = int(abs(math.sin(t * 4)) * 3)
        rect_redondeado(canvas, 10-pulse, 15-pulse,
                         CAM_W+pulse, CAM_H+5+pulse, 12, C_ACCENT, 2+pulse)

    # Barra inferior de la cámara
    bar_y = CAM_H + 10
    rect_redondeado(canvas, 5, bar_y, CAM_W+5, bar_y+35, 8, C_PANEL)
    n_caras = len(caras_info)
    estado_txt = f"{'Registrando...' if estado.modo=='registro' else 'En vivo'}  |  {n_caras} cara(s) detectada(s)"
    cv2.putText(canvas, estado_txt, (18, bar_y+22), F_BODY, 0.42, C_SUBTEXT, 1, cv2.LINE_AA)

    # FPS
    fps_txt = f"FPS: {estado.fps:.0f}" if hasattr(estado, 'fps') else ""
    cv2.putText(canvas, fps_txt, (CAM_W-70, bar_y+22), F_BODY, 0.42, C_SUBTEXT, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════
#  MENSAJE TEMPORAL (TOAST)
# ══════════════════════════════════════════════════════════════
def dibujar_toast(canvas, estado):
    if not estado.msg:
        return
    elapsed = time.time() - estado.msg_tiempo
    if elapsed > 3.0:
        estado.msg = ""
        return
    alpha = min(1.0, (3.0 - elapsed) / 0.5)   # fade out en los últimos 0.5s
    (tw, th), _ = cv2.getTextSize(estado.msg, F_BODY, 0.55, 1)
    cx = WIN_W // 2
    cy = WIN_H - 30
    overlay = canvas.copy()
    rect_redondeado(overlay, cx-tw//2-16, cy-th-10, cx+tw//2+16, cy+10, 10, estado.msg_color)
    cv2.addWeighted(overlay, alpha*0.9, canvas, 1-alpha*0.9, 0, canvas)
    cv2.putText(canvas, estado.msg, (cx-tw//2, cy), F_BODY, 0.55, BLANCO, 1, cv2.LINE_AA)

def mostrar_msg(texto, color=None):
    estado.msg       = texto
    estado.msg_tiempo= time.time()
    estado.msg_color = color or C_ACCENT


# ══════════════════════════════════════════════════════════════
#  CALLBACKS DE MOUSE
# ══════════════════════════════════════════════════════════════
def on_mouse(event, mx, my, flags, param):
    estado.hover_btn = None
    t = time.time()

    # Detectar hover
    if estado.modo == "normal":
        if punto_en_boton(mx, my, BOTONES["registrar"]):
            estado.hover_btn = "registrar"
        elif punto_en_boton(mx, my, BOTONES["eliminar"]):
            estado.hover_btn = "eliminar"
    elif estado.modo == "registro":
        if punto_en_boton(mx, my, BTN_REG_OK):
            estado.hover_btn = "reg_ok"
        elif punto_en_boton(mx, my, BTN_REG_CANCEL):
            estado.hover_btn = "reg_cancel"

    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if t - estado.ultimo_click < 0.25:   # anti-doble-click
        return
    estado.ultimo_click = t

    # ── Clicks en modo NORMAL ──────────────────────────────
    if estado.modo == "normal":
        if punto_en_boton(mx, my, BOTONES["registrar"]):
            estado.modo          = "registro"
            estado.nombre_input  = ""
            estado.progreso_reg  = 0
            estado.input_activo  = True

        elif punto_en_boton(mx, my, BOTONES["eliminar"]):
            personas = param["personas"]
            if personas:
                ultimo = list(personas.keys())[-1]
                del personas[ultimo]
                guardar_personas(personas)
                mostrar_msg(f"'{ultimo}' eliminado", C_DANGER)
            else:
                mostrar_msg("No hay personas registradas", C_DANGER)

    # ── Clicks en modo REGISTRO ────────────────────────────
    elif estado.modo == "registro":
        # Click en input box
        px = CAM_W + 20
        pw = WIN_W - px - 10
        if px+8 <= mx <= WIN_W-18 and 122 <= my <= 158:
            estado.input_activo = True

        elif punto_en_boton(mx, my, BTN_REG_OK):
            if len(estado.nombre_input.strip()) >= 2:
                estado.capturando = True   # señal para el bucle principal
            else:
                mostrar_msg("Escribe al menos 2 caracteres", C_ACCENT2)

        elif punto_en_boton(mx, my, BTN_REG_CANCEL):
            estado.modo         = "normal"
            estado.nombre_input = ""
            estado.progreso_reg = 0
            estado.input_activo = False
            mostrar_msg("Registro cancelado", C_DANGER)


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("Iniciando Face ID GUI...")
    detector = cargar_detector()
    personas = cargar_personas()
    estado.capturando    = False
    estado.desc_buffer   = []
    estado.ultimo_cap_t  = 0.0
    estado.fps           = 0.0
    estado.fps_times     = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" No se pudo abrir la cámara.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cv2.namedWindow("Face ID", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face ID", WIN_W, WIN_H)
    cv2.setMouseCallback("Face ID", on_mouse, {"personas": personas})

    if personas:
        mostrar_msg(f"{len(personas)} persona(s) cargadas", C_ACCENT)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        # ── FPS ──────────────────────────────────────────────
        now = time.time()
        estado.fps_times = [t for t in estado.fps_times if now - t < 1.0]
        estado.fps_times.append(now)
        estado.fps = len(estado.fps_times)

        # ── Detección de caras ────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        caras = detectar_caras(gray, detector)

        # Escalar coordenadas al área de cámara en el canvas
        fh, fw = frame.shape[:2]
        sx = (CAM_W - 10) / fw
        sy = (CAM_H - 10) / fh

        caras_info = []
        for (x, y, w, h) in caras:
            face_gray  = gray[y:y+h, x:x+w]
            descriptor = extraer_descriptor(face_gray)

            if estado.modo == "normal" and personas:
                nombre, confianza = identificar(descriptor, personas)
            else:
                nombre, confianza = "Desconocido", 0.0

            caras_info.append({
                "bbox_scaled": (int(x*sx), int(y*sy), int(w*sx), int(h*sy)),
                "nombre":      nombre,
                "confianza":   confianza,
                "descriptor":  descriptor,
            })

        # ── Captura de registro ───────────────────────────────
        if estado.capturando and estado.progreso_reg < FOTOS_REG:
            if caras_info and (now - estado.ultimo_cap_t) >= 0.7:
                desc = caras_info[0]["descriptor"]
                estado.desc_buffer.append(desc)
                estado.progreso_reg += 1
                estado.ultimo_cap_t  = now

                # Guardar foto de referencia
                os.makedirs(PHOTOS_DIR, exist_ok=True)
                x0, y0, w0, h0 = caras[0] if len(caras) > 0 else (0,0,50,50)
                foto = frame[y0:y0+h0, x0:x0+w0]
                nombre_limpio = estado.nombre_input.strip().replace(" ", "_")
                cv2.imwrite(f"{PHOTOS_DIR}/{nombre_limpio}_{estado.progreso_reg}.jpg", foto)

            # ¿Terminó la captura?
            if estado.progreso_reg >= FOTOS_REG:
                nombre = estado.nombre_input.strip()
                if nombre in personas:
                    personas[nombre].extend(estado.desc_buffer)
                else:
                    personas[nombre] = estado.desc_buffer
                guardar_personas(personas)
                mostrar_msg(f"✓ '{nombre}' registrado con {FOTOS_REG} muestras")
                estado.modo         = "normal"
                estado.capturando   = False
                estado.desc_buffer  = []
                estado.progreso_reg = 0
                estado.nombre_input = ""
                estado.input_activo = False

        # ── Construir canvas ──────────────────────────────────
        canvas = np.full((WIN_H, WIN_W, 3), C_BG, dtype=np.uint8)
        dibujar_area_camara(canvas, frame, caras_info, estado)
        dibujar_panel(canvas, personas, estado)
        dibujar_toast(canvas, estado)

        cv2.imshow("Face ID", canvas)

        # ── Teclado ───────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):   # ESC o Q → salir
            break

        elif estado.input_activo and estado.modo == "registro":
            if key == 13:                  # Enter → confirmar nombre
                if len(estado.nombre_input.strip()) >= 2:
                    estado.capturando = True
                else:
                    mostrar_msg("Escribe al menos 2 caracteres", C_ACCENT2)
            elif key == 8:                 # Backspace
                estado.nombre_input = estado.nombre_input[:-1]
            elif 32 <= key <= 126:         # Caracteres imprimibles
                if len(estado.nombre_input) < 20:
                    estado.nombre_input += chr(key)

    cap.release()
    cv2.destroyAllWindows()
    print("Cerrado correctamente.")


if __name__ == "__main__":
    main()