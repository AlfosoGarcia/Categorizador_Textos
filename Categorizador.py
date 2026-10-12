import os
import pandas as pd
from pypdf import PdfReader
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. PREPARACIÓN DE RECURSOS DE PROCESAMIENTO DE LENGUAJE NATURAL (NLP)


# Se verifica y descarga el corpus de 'stopwords' (palabras sin carga semántica)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Se obtiene la lista de palabras vacías en español
stopwords_espanol = stopwords.words('spanish')

# 2. FUNCIONES DE EXTRACCIÓN Y PROCESAMIENTO DE DATOS

def extraer_texto_pdf(ruta_pdf: str) -> str:
    texto = ""
    try:
        reader = PdfReader(ruta_pdf)
        # Recorrido página por página para extraer el texto
        for pagina in reader.pages:
            contenido = pagina.extract_text()
            if contenido:
                texto += contenido + " "
    except Exception as e:
        print(f"Error al leer el archivo {ruta_pdf}: {e}")

    return texto.strip()


def cargar_dataset_entrenamiento(carpeta_raiz: str) -> tuple[list, list]:
    textos = []
    categorias = []

    # Revisa cada elemento dentro de la carpeta raíz
    for categoria in os.listdir(carpeta_raiz):
        ruta_categoria = os.path.join(carpeta_raiz, categoria)

        # Procesa solo si es un directorio (categoría)
        if os.path.isdir(ruta_categoria):
            for archivo in os.listdir(ruta_categoria):
                # Filtro por extensión PDF (sin importar mayúsculas/minúsculas)
                if archivo.lower().endswith(".pdf"):
                    ruta_pdf = os.path.join(ruta_categoria, archivo)
                    texto_extraido = extraer_texto_pdf(ruta_pdf)

                    # Se almacenan únicamente textos legibles
                    if texto_extraido:
                        textos.append(texto_extraido)
                        categorias.append(categoria)

    return textos, categorias


# 3. FLUJO PRINCIPAL Y ENTRENAMIENTO DEL MODELO


def ejecutar_clasificador():
    # Rutas relativas del proyecto
    carpeta_entrenamiento = "Datos_Entrenamiento"
    carpeta_pruebas = "Datos_Pruebas"

    print("1. Cargando y procesando PDFs de entrenamiento...")
    X_train, y_train = cargar_dataset_entrenamiento(carpeta_entrenamiento)

    # Control de excepciones si la carpeta está vacía
    if not X_train:
        print(
            f"No se encontraron PDFs en '{carpeta_entrenamiento}'. Revisa las carpetas."
        )
        return

    # Construcción del Pipeline de Aprendizaje Automático
    # TfidfVectorizer: Pondera frecuencia de palabras descartando stopwords.
    # MultinomialNB: Algoritmo probabilístico óptimo para clasificación de texto.
    modelo = make_pipeline(
        TfidfVectorizer(stop_words=stopwords_espanol, lowercase=True),
        MultinomialNB(),
    )

    print("2. Entrenando el modelo de Minería de Datos...")
    modelo.fit(X_train, y_train)
    print("¡Modelo entrenado exitosamente!")

    # Evaluation de la carpeta de pruebas
    print("\n3. Clasificando PDFs de la carpeta de pruebas...\n")
    resultados = []

    if not os.path.exists(carpeta_pruebas):
        print(f"La carpeta '{carpeta_pruebas}' no existe.")
        return

    archivos_prueba = [
        f for f in os.listdir(carpeta_pruebas) if f.lower().endswith(".pdf")
    ]

    for archivo in archivos_prueba:
        ruta_pdf = os.path.join(carpeta_pruebas, archivo)
        texto = extraer_texto_pdf(ruta_pdf)

        if texto:
            # Predicción de la categoría con mayor probabilidad
            categoria_predicha = modelo.predict([texto])[0]

            # Cálculo de la probabilidad estimada (nivel de confianza)
            probabilidades = modelo.predict_proba([texto])[0]
            confianza = max(probabilidades) * 100

            resultados.append({
                "Archivo": archivo,
                "Categoría Asignada": categoria_predicha,
                "Confianza (%)": round(confianza, 2),
            })
        else:
            # Manejo de archivos PDF que son imágenes o están corruptos
            resultados.append({
                "Archivo": archivo,
                "Categoría Asignada": "No legible / Vacío",
                "Confianza (%)": 0.0,
            })

    # Presentación final en formato tabla
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados.to_string(index=False))


if __name__ == "__main__":
    ejecutar_clasificador()