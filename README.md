# v2 – Imagen + Video + Cámara (YOLO + Forma + Color + MySQL)

- **Imagen**: `POST /detectar`
- **Video**: `POST /detectar_video` (muestrea ~cada 0.5s, guarda frames anotados y registra cada detección)
- **Cámara**: Live detect en el navegador (envía frames periódicos a `/detectar`, dibuja cajas y guarda en BD).

## Correr
```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate
pip install -r requirements.txt

# (si no lo hiciste) crear BD
mysql -u root -p < schema.sql

# variables (si tu clave no es "password")
# PowerShell:
$env:DB_HOST="127.0.0.1"; $env:DB_USER="root"; $env:DB_PASS="TU_CLAVE"; $env:DB_NAME="detecciones_db"

python app.py
# http://127.0.0.1:8000  (health: /health)
```

## Notas
- Cada detección se inserta como una fila con `objeto, forma, color, fecha, imagen_url`.
- En **imagen** y **cámara**, además se guarda una versión **anotada** con cajas y etiquetas.
- En **video**, se guardan algunos **frames anotados** (los muestreados).

## ngrok
```bash
ngrok config add-authtoken TU_TOKEN
ngrok http 8000
```
