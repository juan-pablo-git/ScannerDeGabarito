import cv2
import numpy as np
import os

def normalizar_resolucao(imagem, altura_padrao=1500):
    h, w = imagem.shape[:2]
    if h == 0:
        raise ValueError("Altura da imagem é 0")
    escala = altura_padrao / h
    imagem_norm = cv2.resize(imagem, None, fx=escala, fy=escala)
    return imagem_norm, escala

def ret_margen(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("debug_thresh.png", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quadrados = []

    h, w = imagem.shape[:2]
    area_img = h * w

    area_min = area_img * 0.00002
    area_max = area_img * 0.0004

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area_min < area < area_max:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, w2, h2 = cv2.boundingRect(approx)
                if 0.7 < (w2 / (h2+1e-6)) < 1.3:
                    quadrados.append((x, y, w2, h2))

    # desenhar contornos detectados para debug
    debug = imagem.copy()
    for q in quadrados:
        x,y,ww,hh = q
        cv2.rectangle(debug, (x,y), (x+ww, y+hh), (0,255,0), 2)
    cv2.imwrite("debug_contours.png", debug)
    return quadrados, contours

def detectar_opcoes(imagem_gabarito, salvar_debug=True):
    if imagem_gabarito is None or imagem_gabarito.size == 0:
        raise ValueError("imagem_gabarito vazia no detectar_opcoes")

    gray = cv2.cvtColor(imagem_gabarito, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if salvar_debug:
        cv2.imwrite("debug_thresh_gabarito.png", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bolhas_detectadas = []

    h, w = imagem_gabarito.shape[:2]
    area_img = h * w
    bolha_min = area_img * 0.0003
    bolha_max = area_img * 0.0030

    for cnt in contours:
        x,y,w2,h2 = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if bolha_min < area < bolha_max:
            aspect_ratio = w2 / (h2+1e-6)
            if 0.75 <= aspect_ratio <= 1.3:
                bolha_roi = thresh[y:y+h2, x:x+w2]
                if bolha_roi.size == 0:
                    continue
                pixels_preenchidos = cv2.countNonZero(bolha_roi)
                porcentagem_preenchimento = (pixels_preenchidos / (w2 * h2)) * 100
                LIMIAR_PREENCHIMENTO = 30
                marcada = porcentagem_preenchimento > LIMIAR_PREENCHIMENTO
                bolhas_detectadas.append({
                    'center_x': int(x + w2/2),
                    'center_y': int(y + h2/2),
                    'w': int(w2),
                    'h': int(h2),
                    'marcada': bool(marcada),
                    'area': float(area),
                    'porcentagem': float(porcentagem_preenchimento)
                })
                cor = (0,255,0) if marcada else (0,0,255)
                cv2.rectangle(imagem_gabarito, (x,y), (x+w2, y+h2), cor, 2)

    if salvar_debug:
        cv2.imwrite("debug_gabarito_boxes.png", imagem_gabarito)
    return bolhas_detectadas

def encontrar_maior_contorno_bbox(imagem):
    gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    maior = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(maior, True)
    approx = cv2.approxPolyDP(maior, 0.02*peri, True)
    # se for quadrilátero, usa os pontos; senão usa bounding rect
    if len(approx) >= 4:
        x,y,w,h = cv2.boundingRect(approx)
    else:
        x,y,w,h = cv2.boundingRect(maior)
    debug = imagem.copy()
    cv2.rectangle(debug, (x,y), (x+w,y+h), (255,0,0), 3)
    cv2.imwrite("debug_sheet.png", debug)
    return (x,y,w,h)

def clamp_int(v, maxi):
    return max(0, min(int(v), int(maxi)))

# ---------------- main ----------------
caminho = "gabarito.jpg"
if not os.path.exists(caminho):
    print("Arquivo", caminho, "não encontrado.")
    exit()

imagem = cv2.imread(caminho)
print("imagem raw:", None if imagem is None else imagem.shape)

if imagem is None:
    print("Erro: não consegui abrir a imagem.")
    exit()

imagem, escala = normalizar_resolucao(imagem)
print("imagem normalizada:", imagem.shape, "escala:", escala)

quadrados, contours = ret_margen(imagem)
print("quadrados detectados:", len(quadrados), quadrados)

# Se tem pelo menos 4, usa método dos 4 cantos
if len(quadrados) >= 4:
    # ordenar e calcular box que engloba os 4
    quadrados = sorted(quadrados, key=lambda x: (x[1], x[0]))
    xs = [q[0] for q in quadrados]
    ys = [q[1] for q in quadrados]
    ws = [q[2] for q in quadrados]
    hs = [q[3] for q in quadrados]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max([xs[i] + ws[i] for i in range(len(quadrados))])
    y2 = max([ys[i] + hs[i] for i in range(len(quadrados))])
    # clamp
    x1 = clamp_int(x1, imagem.shape[1]-1)
    y1 = clamp_int(y1, imagem.shape[0]-1)
    x2 = clamp_int(x2, imagem.shape[1])
    y2 = clamp_int(y2, imagem.shape[0])
    imagemGabarito = imagem[y1:y2, x1:x2]
    print("recorte pelos 4 quadrados:", (x1,y1,x2,y2), "gabarito shape:", None if imagemGabarito is None else imagemGabarito.shape)
else:
    # fallback: tenta encontrar o maior contorno (a folha)
    print("Menos que 4 quadrados. Tentando maior contorno como folha...")
    bbox = encontrar_maior_contorno_bbox(imagem)
    if bbox is not None:
        x,y,w,h = bbox
        x1 = clamp_int(x, imagem.shape[1]-1)
        y1 = clamp_int(y, imagem.shape[0]-1)
        x2 = clamp_int(x + w, imagem.shape[1])
        y2 = clamp_int(y + h, imagem.shape[0])
        imagemGabarito = imagem[y1:y2, x1:x2]
        print("recorte pelo maior contorno:", (x1,y1,x2,y2), "gabarito shape:", None if imagemGabarito is None else imagemGabarito.shape)
    else:
        # último recurso: usa imagem inteira
        print("Não achei contorno. Usando a imagem inteira como gabarito.")
        imagemGabarito = imagem.copy()
        print("gabarito shape (total):", imagemGabarito.shape)

# Verificações finais
if imagemGabarito is None or imagemGabarito.size == 0:
    print("ERRO FINAL: imagemGabarito está vazia. Saindo.")
    exit()

try:
    resultados = detectar_opcoes(imagemGabarito)
    print("Resultados:", resultados)
except Exception as e:
    print("Erro ao detectar opcoes:", e)
    exit()

# Mostrar (opcional)
cv2.imshow("Gabarito", imagemGabarito)
cv2.waitKey(0)
cv2.destroyAllWindows()
