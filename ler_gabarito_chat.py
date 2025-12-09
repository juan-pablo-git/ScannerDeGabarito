import cv2
import numpy as np

# Carregar imagem
img = cv2.imread("gabarito.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

print(f"Quadrados detectados: {len(quadrados)}")
cv2.imshow("Quadrados detectados", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
