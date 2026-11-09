import os
import traceback
 
import numpy as np
import tensorflow as tf
import matplotlib
 
matplotlib.use("Agg")  # guardar la figura en un archivo, sin abrir ventanas
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
from scipy import ndimage
 
RUTA_MODELO = "modelo_mnist.keras"
EPOCAS = 30
BATCH_SIZE = 32
 
 
# ------------------------------------------------------------
# 1. Datos
# ------------------------------------------------------------
def cargar_datos():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
 
    # Guardar un ejemplo del dataset como imagen
    plt.imshow(x_train[0].reshape(28, 28), cmap="gray")
    plt.title(f"Ejemplo de MNIST (etiqueta: {y_train[0]})")
    plt.savefig("ejemplo_mnist.png")
    plt.close()
 
    return (x_train, y_train), (x_test, y_test)
 
 
# ------------------------------------------------------------
# 2. Modelo
# ------------------------------------------------------------
def crear_modelo():
    entrada = tf.keras.layers.Input(shape=(784,))
    capa1 = tf.keras.layers.Dense(64, activation="relu")(entrada)
    # ReLU: deja en 0 los valores negativos
    capa2 = tf.keras.layers.Dense(16, activation="relu")(capa1)
    # Softmax: devuelve un vector de 10 probabilidades (una por dígito).
    # Muestra la "seguridad" del modelo, pero eso no garantiza que sea confiable.
    salida = tf.keras.layers.Dense(10, activation="softmax")(capa2)
    return tf.keras.Model(inputs=entrada, outputs=salida)
 
 
def obtener_modelo():
    (x_train, y_train), (x_test, y_test) = cargar_datos()
 
    if os.path.exists(RUTA_MODELO):
        print(f"Cargando modelo guardado: {RUTA_MODELO}")
        modelo = tf.keras.models.load_model(RUTA_MODELO)
    else:
        modelo = crear_modelo()
        modelo.summary()
        modelo.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        modelo.fit(x_train, y_train, epochs=EPOCAS, batch_size=BATCH_SIZE)
        modelo.save(RUTA_MODELO)
        print(f"Modelo guardado en {RUTA_MODELO}")
 
    # Evaluación sobre datos que el modelo nunca vio durante el entrenamiento
    perdida, precision = modelo.evaluate(x_test, y_test, verbose=0)
    print(f"Precisión en prueba: {precision:.4f}  (loss: {perdida:.4f})")
    return modelo
 
 
# ------------------------------------------------------------
# 3. Preprocesamiento de la imagen del usuario
# ------------------------------------------------------------
def preprocesar(imagen):
    """Convierte una foto/imagen de un dígito al formato de MNIST:
    28x28, fondo negro, trazo blanco de ~2-3 px, dígito en caja de 20x20
    y centrado por centro de masa. Devuelve un array 28x28 con valores 0-255."""
 
    # Gradio puede entregar un dict (editor) o un array numpy
    if isinstance(imagen, dict):
        imagen = imagen.get("composite", imagen.get("background"))
 
    img = Image.fromarray(np.asarray(imagen).astype("uint8"))
 
    # Si tiene transparencia, ponerla sobre fondo blanco
    if img.mode in ("RGBA", "LA"):
        fondo = Image.new("RGB", img.size, (255, 255, 255))
        fondo.paste(img, mask=img.split()[-1])
        img = fondo
 
    arr = np.asarray(img.convert("L")).astype("float32")
 
    # Si el fondo es claro (hoja blanca), invertir: MNIST es trazo blanco sobre negro
    borde = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.median(borde) > 127:
        arr = 255 - arr
 
    # Quitar el fondo (papel, cuadrícula, sombras suaves)
    arr = arr - np.percentile(arr, 50)
    arr[arr < 0] = 0
    if arr.max() > 0:
        arr = arr / arr.max() * 255
    arr[arr < 60] = 0  # elimina cuadrícula y ruido débil
 
    # Recortar al rectángulo que contiene el dígito
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        return np.zeros((28, 28), dtype="uint8")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
 
    # Escalar para que el lado mayor mida 20 px (como en MNIST)
    h, w = arr.shape
    escala = 20.0 / max(h, w)
    nuevo_w, nuevo_h = max(1, int(round(w * escala))), max(1, int(round(h * escala)))
    arr = np.asarray(
        Image.fromarray(arr.astype("uint8")).resize((nuevo_w, nuevo_h), Image.LANCZOS)
    ).astype("float32")
 
    # Al reducir, el trazo se vuelve tenue: volver a llevarlo a 0-255
    if arr.max() > 0:
        arr = arr / arr.max() * 255
 
    # Si el trazo quedó muy fino (< ~2.5 px), engrosarlo como en MNIST
    mascara = arr > 127
    if mascara.any():
        dist = ndimage.distance_transform_edt(mascara)
        grosor = 2 * np.percentile(dist[mascara], 95)
        if grosor < 2.5:
            arr = ndimage.grey_dilation(arr, size=(3, 3))
            arr = ndimage.gaussian_filter(arr, 0.6)
            arr = arr / arr.max() * 255
 
    # Pegar en un lienzo 28x28
    lienzo = np.zeros((28, 28), dtype="float32")
    y0, x0 = (28 - nuevo_h) // 2, (28 - nuevo_w) // 2
    lienzo[y0:y0 + nuevo_h, x0:x0 + nuevo_w] = arr
 
    # Centrar por centro de masa (así están centrados los dígitos de MNIST)
    cy, cx = ndimage.center_of_mass(lienzo)
    lienzo = ndimage.shift(lienzo, (14 - cy, 14 - cx), order=1, mode="constant")
 
    return np.clip(lienzo, 0, 255).astype("uint8")
 
 
# ------------------------------------------------------------
# 4. Interfaz
# ------------------------------------------------------------
def crear_interfaz(modelo):
    def predecir(imagen):
        if imagen is None:
            return None, None
        try:
            img28 = preprocesar(imagen)
 
            # El modelo se entrenó SIN dividir entre 255,
            # así que la entrada se queda en el rango 0-255.
            entrada = img28.reshape(1, 784).astype("float32")
            probs = modelo.predict(entrada, verbose=0)[0]
 
            confianzas = {str(i): float(probs[i]) for i in range(10)}
 
            # Ampliar 10x solo para verlo mejor (el modelo recibe el de 28x28)
            vista = np.kron(img28, np.ones((10, 10), dtype="uint8"))
            return confianzas, vista
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(f"{type(e).__name__}: {e}")
 
    return gr.Interface(
        fn=predecir,
        inputs=gr.Image(label="Sube la imagen de tu número escrito a mano", type="numpy"),
        outputs=[
            gr.Label(num_top_classes=10, label="Predicción del modelo (probabilidad por clase)"),
            gr.Image(label="Lo que ve el modelo (28x28, ampliado para verlo)"),
        ],
        title="Reconocedor de dígitos escritos a mano (MNIST)",
        description=(
            "Red densa 784 → 64 → 16 → 10. Sube una foto de un solo dígito, "
            "preferiblemente con trazo grueso y oscuro sobre fondo claro."
        ),
    )
 
 
# ------------------------------------------------------------
# 5. Programa principal
# ------------------------------------------------------------
if __name__ == "__main__":
    modelo = obtener_modelo()
    interfaz = crear_interfaz(modelo)
    interfaz.launch(inbrowser=True)  # abre http://127.0.0.1:7860 en el navegador