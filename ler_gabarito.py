import cv2

# Caminho da imagem
caminho = "gabarito.jpg"  # troque pelo nome do arquivo
imagem = cv2.imread(caminho)

def ret_margen(imagem):
            # Carregar imagem
        gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

        # Binarizar (preto/branco)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        quadrados = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filtro por área (ajuste conforme necessário)
            if 500 < area < 5000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

                # É quadrado se tiver 4 lados
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)

                    # Quadrado: largura ≈ altura
                    if 0.85 < w/h < 1.15:
                        quadrados.append((x, y, w, h))
                        # cv2.rectangle(imagem, (x,y), (x+w, y+h), (0,255,0), 2)
        # cv2.imshow("Quadrados detectados", imagem)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return quadrados
def recize_image(imagem, scale): 
    width = int(imagem.shape[1] * scale / 100)
    height = int(imagem.shape[0] * scale / 100)
    imagemResize = cv2.resize(imagem, (width, height))  # Redimensiona para facilitar a visualização 
    return imagemResize

def detectar_opcoes(imagem_gabarito):
    """
    Processa a imagem recortada do gabarito para detectar e avaliar as bolhas.
    """
    
    # 1. PRÉ-PROCESSAMENTO 🧼
    
    # Converte para tons de cinza
    gray = cv2.cvtColor(imagem_gabarito, cv2.COLOR_BGR2GRAY)
    
    # Aplicar Gaussian Blur para reduzir ruído e suavizar as bordas (essencial para thresholding)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Binarização (Threshold): Converte a imagem em preto e branco.
    # O valor 0 (preto) representa as bolhas preenchidas/texto, 255 (branco) o fundo.
    # Usamos cv2.THRESH_OTSU para deixar o OpenCV calcular o melhor limiar automaticamente.
    # cv2.THRESH_BINARY_INV: Inverte para que as áreas preenchidas fiquem BRANCAS.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Se precisar visualizar o resultado do threshold:
    # cv2.imshow("Threshold", cv2.resize(thresh, (int(thresh.shape[1]*0.8), int(thresh.shape[0]*0.8))))
    # cv2.waitKey(0)
    
    # 2. DETECÇÃO DE CONTORNOS (As bolhas) ⭕
    
    # Encontra contornos na imagem binarizada
    # cv2.RETR_EXTERNAL: Apenas contornos externos (evita contornos internos de texto, etc.)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bolhas_detectadas = []
    
    # Loop em todos os contornos encontrados
    for i, cnt in enumerate(contours):
        # Calcula o Bounding Box (retângulo envolvente)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Área do contorno
        area = cv2.contourArea(cnt)
        
        # Filtros: Tentativa de isolar SOMENTE as bolhas de resposta
        # Ajuste os valores (área, proporção) de acordo com o seu gabarito.
        if 800 < area < 3500:  # Filtro por área razoável para uma bolha
            # Filtro por circularidade (largura e altura devem ser semelhantes)
            aspect_ratio = w / h
            if 0.9 <= aspect_ratio <= 1.1:
                
                # 3. AVALIAÇÃO DA BOLHA (Preenchida ou Vazia) 🖍️
                
                # Recorta a região de interesse (ROI) do contorno na imagem Binarizada
                bolha_roi = thresh[y:y+h, x:x+w]
                
                # Calcula a área preenchida dentro da ROI (contagem de pixels brancos)
                # O threshold INV garantiu que a área preenchida dentro da bolha seja branca (255)
                pixels_preenchidos = cv2.countNonZero(bolha_roi)
                
                # A porcentagem de preenchimento é crucial para determinar se a bolha foi marcada.
                # Área total do bounding box é w * h
                porcentagem_preenchimento = (pixels_preenchidos / (w * h)) * 100
                
                # **LIMIAR DE MARCAÇÃO:** Ajuste este valor (ex: 30%)
                # Bolhas preenchidas terão uma % de preenchimento maior.
                LIMIAR_PREENCHIMENTO = 30 
                
                marcada = porcentagem_preenchimento > LIMIAR_PREENCHIMENTO
                
                bolhas_detectadas.append({
                    'center_x': x + w // 2,  # Centro X
                    'center_y': y + h // 2,  # Centro Y
                    'w': w,
                    'h': h,
                    'marcada': marcada,
                    'area': area,
                    'porcentagem': porcentagem_preenchimento
                })
                
                # Desenha um retângulo de debug (verde se marcada, vermelho se vazia)
                cor = (0, 255, 0) if marcada else (0, 0, 255) # BGR
                cv2.rectangle(imagem_gabarito, (x, y), (x + w, y + h), cor, 2)

    return bolhas_detectadas

# Verifica se carregou
imagemPadronizada = recize_image(imagem, 30)
margen = ret_margen(imagem)
imagemGabarito = imagem[margen[3][1]:margen[0][1],margen[3][0]:margen[0][0]]
print(detectar_opcoes(imagemGabarito))
# Lê a imagem

if imagem is None:
    print("Erro: não consegui abrir a imagem.")
    exit()

# Exibe
cv2.imshow("Imagem carregada",  imagemGabarito)
cv2.waitKey(0)
cv2.destroyAllWindows()
