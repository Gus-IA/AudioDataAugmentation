# Audio Data Augmentation con PyTorch y Torchaudio

Este proyecto demuestra cómo agregar ruido a señales de audio usando PyTorch y Torchaudio, y cómo reproducir los resultados con `sounddevice`.

## Contenido del proyecto

- Descarga de audios de ejemplo (`wav` y `RIR`) desde los tutoriales de torchaudio.
- Normalización de audio para mantener amplitudes en el rango [-1, 1].
- Aplicación de ruido a la señal de voz original con diferentes **SNRs** (20, 10 y 3 dB).
- Preparación del audio para reproducción con `sounddevice`.


🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
