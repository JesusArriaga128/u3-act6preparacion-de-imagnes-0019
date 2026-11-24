import cv2
import numpy as np
import os
import csv
## Jesus 
# 0019

# 1. Configuración inicial
RUTA_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

carpeta_original = os.path.join(RUTA_BASE, "img-original-0019")
# Si existe la carpeta intermedia 'Manzanas', entra automáticamente a ella
if os.path.exists(os.path.join(carpeta_original, "Manzanas", "peras")):
    carpeta_original = os.path.join(carpeta_original, "Manzanas")

carpeta_preparadas = os.path.join(RUTA_BASE, "img-preparadas-0019")
carpeta_resultados = os.path.join(RUTA_BASE, "resultados-img-0019")

ANCHO_DESTINO = 224
ALTO_DESTINO = 224

os.makedirs(carpeta_preparadas, exist_ok=True)
os.makedirs(carpeta_resultados, exist_ok=True)

categorias = ["manzanas", "peras"]
reporte = []
imagenes_procesadas = 0
imagenes_aprobadas = 0

print("🚀 Iniciando preparación de imágenes...\n")
print('Jesus Arriaga 0019')

# 2. Recorrer cada categoría y sus archivos
for categoria in categorias:
    ruta_origen = os.path.join(carpeta_original, categoria)
    ruta_destino = os.path.join(carpeta_preparadas, categoria)
    
    os.makedirs(ruta_destino, exist_ok=True)
    
    if not os.path.exists(ruta_origen):
        print(f"⚠️ La carpeta {ruta_origen} no existe. Saltando...")
        continue
        
    for nombre_archivo in os.listdir(ruta_origen):
        if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.jfif')):
            continue
            
        ruta_imagen = os.path.join(ruta_origen, nombre_archivo)
        nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
        
        print(f"📸 Procesando: {nombre_archivo}")
        
        # 3. Leer la imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"   ❌ Error: No se pudo leer la imagen.")
            reporte.append([categoria, nombre_archivo, "ERROR", "-", "-", "-", "-", "-", "-", "REVISAR"])
            continue
            
        imagenes_procesadas += 1
        alto_orig, ancho_orig, canales = imagen.shape
        print(f"   📏 Dimensiones originales: {ancho_orig} x {alto_orig}")
        
        # 4. Redimensionar la imagen
        imagen_redimensionada = cv2.resize(imagen, (ANCHO_DESTINO, ALTO_DESTINO), interpolation=cv2.INTER_AREA)
        print(f"   📐 Dimensiones nuevas: {ANCHO_DESTINO} x {ALTO_DESTINO}")
        
        # Guardar la imagen redimensionada como JPG
        ruta_jpg = os.path.join(ruta_destino, f"{nombre_sin_ext}.jpg")
        cv2.imwrite(ruta_jpg, imagen_redimensionada)
        
        # 5. Normalizar los valores de los píxeles (de 0-255 a 0.0-1.0)
        imagen_normalizada = imagen_redimensionada.astype(np.float32) / 255.0
        val_min = float(np.min(imagen_normalizada))
        val_max = float(np.max(imagen_normalizada))
        print(f"   🔢 Rango normalizado: {val_min:.3f} a {val_max:.3f}")
        
        # Guardar los datos numéricos normalizados como archivo .npy
        ruta_npy = os.path.join(ruta_destino, f"{nombre_sin_ext}.npy")
        np.save(ruta_npy, imagen_normalizada)
        
        # 6. Evaluar automáticamente el resultado
        dimensiones_correctas = (imagen_redimensionada.shape[0] == ALTO_DESTINO and 
                                 imagen_redimensionada.shape[1] == ANCHO_DESTINO)
        rango_correcto = (val_min >= 0.0 and val_max <= 1.0)
        archivos_guardados = os.path.isfile(ruta_jpg) and os.path.isfile(ruta_npy)
        
        if dimensiones_correctas and rango_correcto and archivos_guardados:
            estado = "APROBADA"
            imagenes_aprobadas += 1
            print(f"   ✅ Evaluación: APROBADA\n")
        else:
            estado = "REVISAR"
            print(f"   ⚠️ Evaluación: REVISAR\n")
            
        # 7. Guardar datos en la lista para el reporte final
        reporte.append([
            categoria, nombre_archivo, "OK", ancho_orig, alto_orig,
            ANCHO_DESTINO, ALTO_DESTINO, round(val_min, 3), round(val_max, 3), estado
        ])

# 8. Generar el reporte en formato CSV
ruta_reporte = os.path.join(carpeta_resultados, "reporte_preparacion.csv")
with open(ruta_reporte, "w", newline="", encoding="utf-8") as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow([
        "Categoría", "Archivo original", "Lectura", 
        "Ancho original", "Alto original", "Ancho preparado", "Alto preparado", 
        "Valor mínimo norm.", "Valor máximo norm.", "Evaluación"
    ])
    escritor.writerows(reporte)

# 9. Resumen final en consola
print("=" * 45)
print("🎉 RESUMEN DE LA PRÁCTICA")
print("=" * 45)
print(f"Imágenes procesadas: {imagenes_procesadas}")
print(f"Imágenes aprobadas:  {imagenes_aprobadas}")
print(f"Imágenes a revisar:  {imagenes_procesadas - imagenes_aprobadas}")
print(f"\n📁 Reporte guardado en: {ruta_reporte}")
print("✅ PRÁCTICA TERMINADA EXITOSAMENTE")
print('Progra realizado por Jesus 0019')











# --- BLOQUE PARA MOSTRAR Y GUARDAR MOSAICO POR CATEGORÍA ---
print("\n🖼️ Generando vista conjunta por categoría...")

for categoria in categorias:
    ruta_cat = os.path.join(carpeta_preparadas, categoria)
    if not os.path.exists(ruta_cat):
        continue

    imagenes_fila = []
    # Leer solo los .jpg ya redimensionados (224x224)
    archivos_jpg = sorted([f for f in os.listdir(ruta_cat) if f.lower().endswith('.jpg')])

    for archivo in archivos_jpg:
        img = cv2.imread(os.path.join(ruta_cat, archivo))
        if img is not None:
            # Asegurar que todas tengan exactamente 224x224
            img = cv2.resize(img, (ANCHO_DESTINO, ALTO_DESTINO))
            imagenes_fila.append(img)

    if imagenes_fila:
        # Unir todas las imágenes de la categoría en una sola fila horizontal
        mosaico = np.hstack(imagenes_fila)

        # Guardar el mosaico como evidencia en resultados
        ruta_mosaico = os.path.join(carpeta_resultados, f"mosaico_{categoria}.jpg")
        cv2.imwrite(ruta_mosaico, mosaico)
        print(f"   ✅ Mosaico guardado en: {ruta_mosaico}")

        # Mostrar en una ventana emergente
        cv2.imshow(f"Categoria: {categoria.upper()} (224x224 c/u)", mosaico)

print("\n👀 Presiona cualquier tecla sobre las ventanas de imagen para cerrarlas...")
cv2.waitKey(0)
cv2.destroyAllWindows()