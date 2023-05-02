from flask import Flask, render_template
import yfinance as yf
import datetime
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route("/")
def index():
    try:
        # Descargar datos históricos de BTC-ETH
        fecha_inicio = datetime.datetime(2022, 1, 1)
        fecha_fin = datetime.datetime.now()
        data = yf.download('BTC-ETH', period='1d', start=fecha_inicio, end=fecha_fin)
        data_bit=yf.download('BTC-USD',period="30m",start=fecha_inicio,end=fecha_fin)
        data_eth=yf.download('ETH-USD',period="30m",start=fecha_inicio,end=fecha_fin)
        cierre = data['Close']
        cierre_bit=data_bit['Close']
        cierre_eth=data_eth['Close']

        datos = cierre.values
        datos_bit=cierre_bit.values
        datos_eth=cierre_eth.values

        # Crear conjunto de objetivos
        objetivos = datos[1:]
        objetivos = tf.reshape(objetivos, shape=(len(objetivos), 1))

        # Crear modelo
        oculta1 = tf.keras.layers.Dense(units=10, input_shape=[1])
        oculta2 = tf.keras.layers.Dense(units=10)
        salida = tf.keras.layers.Dense(units=1)
        modelo = tf.keras.Sequential([oculta1, oculta2, salida])
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='mean_squared_error'
        )

        # Entrenar modelo
        print("Comenzando entrenamiento...")
        historial = modelo.fit(datos[1:], objetivos, epochs=17, verbose=False)
        print("Modelo entrenado!")

        
        # Hacer una predicción para mañana
        fecha_prediccion = datetime.datetime.now()
        datos_prediccion = yf.download('BTC-ETH', period='1m', start=fecha_prediccion, end=fecha_prediccion)
        cierre_prediccion = datos_prediccion['Close'].values

        # Asegurarse de que haya datos para predecir
        if len(cierre_prediccion) > 0:
            prediccion = modelo.predict(np.array([cierre_prediccion[0]]))
            print("La predicción para el valor de BTC-ETH mañana es: ", prediccion[0][0])

            # Crear gráfico
            plt.xlabel("# Epoca")
            plt.ylabel("Magnitud de pérdida")
            plt.plot(historial.history["loss"])
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            buffer_img = base64.b64encode(buf.read()).decode('ascii')
            return render_template("index.html", prediccion=prediccion[0][0], buffer_img=buffer_img,datos_bit=datos_bit,datos_eth=datos_eth,fecha_fin=fecha_fin)
        else:
            print("No hay datos para predecir el valor de BTC-ETH mañana.")
            return render_template("index.html", error="No hay datos para predecir el valor de BTC-ETH mañana.")
    except Exception as e:
        print(e)
        return render_template("index.html", error="Error interno del servidor.")

if __name__ == "main":
    app.run(debug=True)
