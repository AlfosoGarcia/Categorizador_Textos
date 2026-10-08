import os
import pandas as pd
from pypdf import PdfReader
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Descargar las palabras vacías (stopwords) en español si no están descargadas
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Cargar lista de palabras vacías en español
stopwords_espanol = stopwords.words('spanish')

def extraer_texto_pdf(ruta_pdf):
    texto = ""
    try:
        reader = PdfReader(ruta_pdf)
        for pagina in reader.pages:
            contenido = pagina.extract_text()
            if contenido:
                texto += contenido + " "
    except Exception as e:
        print(f"Error al leer {ruta_pdf}: {e}")
    return texto.strip()

def cargar_dataset_entrenamiento(carpeta_raiz):
    textos = []
    categorias = []
    
    for categoria in os.listdir(carpeta_raiz):
        ruta_categoria = os.path.join(carpeta_raiz, categoria)
        if os.path.isdir(ruta_categoria):
            for archivo in os.listdir(ruta_categoria):
                if archivo.lower().endswith(".pdf"):
                    ruta_pdf = os.path.join(ruta_categoria, archivo)
                    texto_extraido = extraer_texto_pdf(ruta_pdf)
                    
                    if texto_extraido:
                        textos.append(texto_extraido)
                        categorias.append(categoria)
                        
    return textos, categorias

def ejecutar_clasificador():
    carpeta_entrenamiento = "Datos_Entrenamiento"
    carpeta_pruebas = "Datos_Pruebas"
    
    print("1. Cargando y procesando PDFs de entrenamiento...")
    X_train, y_train = cargar_dataset_entrenamiento(carpeta_entrenamiento)
    
    if not X_train:
        print(f"No se encontraron PDFs en '{carpeta_entrenamiento}'. Revisa las carpetas.")
        return

    # Usamos la lista de stopwords en español descargada desde NLTK
    modelo = make_pipeline(
        TfidfVectorizer(stop_words=stopwords_espanol, lowercase=True),
        MultinomialNB()
    )

    print("2. Entrenando el modelo de Minería de Datos...")
    modelo.fit(X_train, y_train)
    print("¡Modelo entrenado exitosamente!")

    print("\n3. Clasificando PDFs de la carpeta de pruebas...\n")
    resultados = []

    if not os.path.exists(carpeta_pruebas):
        print(f"La carpeta '{carpeta_pruebas}' no existe.")
        return

    archivos_prueba = [f for f in os.listdir(carpeta_pruebas) if f.lower().endswith(".pdf")]

    for archivo in archivos_prueba:
        ruta_pdf = os.path.join(carpeta_pruebas, archivo)
        texto = extraer_texto_pdf(ruta_pdf)

        if texto:
            categoria_predicha = modelo.predict([texto])[0]
            probabilidades = modelo.predict_proba([texto])[0]
            confianza = max(probabilidades) * 100

            resultados.append({
                "Archivo": archivo,
                "Categoría Asignada": categoria_predicha,
                "Confianza (%)": round(confianza, 2)
            })
        else:
            resultados.append({
                "Archivo": archivo,
                "Categoría Asignada": "No legible / Vacío",
                "Confianza (%)": 0.0
            })

    df_resultados = pd.DataFrame(resultados)
    print(df_resultados.to_string(index=False))

if __name__ == "__main__":
    ejecutar_clasificador()