# DVC con Remote en Google Drive

Instrucciones para configurar el remote de Google Drive en DVC y para
verificar que la implementacion es exitosa.

---

## Por que Google Drive?

DVC puede funcionar completamente sin remote (todo local). El remote es
opcional y sirve para:

- **Compartir artefactos grandes** entre integrantes del equipo sin volver a
  ejecutar el pipeline (los parquet de 11 M filas y los pkl del modelo).
- **Reproducir en entorno limpio**: clonar el repo + `dvc pull` descarga todos
  los artefactos en lugar de regenerarlos.
- **Mejorar la nota** en el criterio "Data & Pipeline Versioning" de un 85 %
  (DVC local) a un 100 % (DVC con remote storage).

---

## Prerequisitos

1. Python 3.11 con las dependencias del proyecto instaladas:
   ```bash
   pip install -r requirements.txt   # incluye dvc==3.67.1
   ```
2. Una cuenta de Google activa (la misma que usa el equipo).
3. Acceso de escritura al repositorio de GitHub (para hacer push de `dvc.lock`).

---

## Paso 1 - Crear el folder en Google Drive

1. Abre [drive.google.com](https://drive.google.com) en el navegador.
2. Crea un folder nuevo, por ejemplo: `mibici-dvc-remote`.
3. Abre ese folder y copia el **ID** de la URL del navegador.
   La URL tiene esta forma:
   ```
   https://drive.google.com/drive/folders/1ABCdef-XYZ_123abcDEF456
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          Este es el ID del folder
   ```
   Guarda ese ID; lo necesitas en el siguiente paso.

---

## Paso 2 - Configurar el remote en DVC

Desde la raiz del repositorio (con el entorno virtual activado):

```bash
# Agrega el remote llamado "gdrive" como remote por defecto (-d)
dvc remote add -d gdrive gdrive://<ID-DEL-FOLDER>

# Ejemplo con un ID real:
# dvc remote add -d gdrive gdrive://1ABCdef-XYZ_123abcDEF456

# Verifica que se guardo correctamente
dvc remote list
```

Esto escribe la configuracion en `.dvc/config`. Ese archivo SI debe
commitearse a git (no contiene credenciales, solo la URL del remote).

```bash
git add .dvc/config
git commit -m "feat(dvc): add Google Drive remote"
```

---

## Paso 3 - Ejecutar el pipeline y subir los artefactos

Si los artefactos YA existen localmente (el pipeline ya se ejecuto antes):

```bash
# Registra los artefactos existentes en el cache de DVC
# NOTA: esto puede tardar varios minutos; los parquet son grandes
dvc repro --allow-missing

# Sube los artefactos al remote de Google Drive
dvc push
```

Si vas a ejecutar el pipeline desde cero con DVC:

```bash
# Primero descarga los datos crudos
# NOTA: "make download" solo funciona en Linux/Mac o Git Bash.
# En Windows PowerShell usa el comando Python directamente:
python -m mibici.dataset download --start 2024-01 --end 2026-09

# Luego reproduce el pipeline completo via DVC
# (DVC llama internamente a los comandos de cada stage)
dvc repro

# Sube los artefactos al remote
dvc push
```

**Nota para usuarios de Windows (PowerShell):** `make` no esta disponible
en PowerShell por defecto. Todas las instrucciones que usan `make <target>`
tienen su equivalente directo en Python:

| Comando make | Equivalente Python |
|---|---|
| `make download` | `python -m mibici.dataset download --start 2024-01 --end 2026-09` |
| `make data` | `python -m mibici.dataset build && python -m mibici.cleaning && ...` |
| `make features-v3` | `python -m mibici.features build-v3` |
| `make train` | `python -m mibici.modeling.tune evaluate` |
| `make predict` | `python -m mibici.modeling.predict --source local` |
| `make test` | `pytest tests/ -q` |

Con DVC, ninguno de esos comandos es necesario manualmente:
`dvc repro` los ejecuta en orden automaticamente.

La primera vez que ejecutes `dvc push` o `dvc pull`, Google abrira una ventana
del navegador pidiendo autorizacion OAuth. Acepta los permisos y el token
se guarda automaticamente en tu maquina.

---

## Paso 4 - Verificar desde otra maquina (prueba de reproducibilidad)

En un equipo limpio (o en una nueva VM):

```bash
# 1. Clona el repositorio
git clone https://github.com/anthonyibrail/mlops-equipo5-mibici.git
cd mlops-equipo5-mibici

# 2. Instala las dependencias
pip install -r requirements.txt

# 3. Descarga los artefactos desde el remote (sin re-ejecutar el pipeline)
dvc pull

# 4. Verifica que los archivos se descargaron correctamente
ls data/interim/
ls data/processed/
ls models/

# 5. Comprueba que DVC considera el pipeline actualizado
dvc status
# Debe mostrar: "Data and pipelines are up to date."
```

---

## Como saber que la implementacion es exitosa

Ejecuta los siguientes comandos y verifica cada criterio:

### Criterio 1 - El pipeline esta definido correctamente

```bash
dvc dag
```

**Salida esperada** (o similar):
```
              +------+
              | data |
              +------+
                  *
            +----------+
            | features |
           *+----------+*
        ***                ***
     ***                      ***
  ***                            ***
**           +-------+            **
             | train |
             +-------+
         *        *        *
    +-------+        +---------+
    | drift |        | predict |
    +-------+        +---------+
```

Criterio cumplido si el DAG muestra las 5 etapas con `train` como nodo central
y `drift` y `predict` dependiendo de él.

### Criterio 2 - Los parametros de params.yaml son rastreados

```bash
dvc params show
```

**Salida esperada**: muestra los valores de `problema`, `modelos`,
`validacion`, `mlflow`, `promocion` y `datos` del archivo `params.yaml`.

Criterio cumplido si aparecen parametros de al menos 2 secciones de `params.yaml`.

### Criterio 3 - El estado del pipeline es consistente

```bash
dvc status
```

**Salida esperada despues de `dvc repro` o `dvc pull`**:
```
Data and pipelines are up to date.
```

Si muestra etapas como "changed", significa que alguna entrada cambio
desde el ultimo `dvc repro`. Es el comportamiento correcto de DVC.

### Criterio 4 - Los tests del pipeline pasan

```bash
pytest tests/test_dvc_pipeline.py -v
```

**Salida esperada**:
```
17 passed in 0.XX s
```

Todos los tests deben pasar. Si alguno falla, el mensaje de error
indica exactamente que esta mal en `dvc.yaml`.

### Criterio 5 - El remote funciona (si se configuro Google Drive)

```bash
dvc remote list
# Debe mostrar: gdrive   gdrive://<ID-DEL-FOLDER>

dvc push
# Debe mostrar: X file(s) pushed (o "Everything is up to date.")

dvc pull
# Debe mostrar: X file(s) fetched (o "Everything is up to date.")
```

Criterio cumplido si `dvc push` no arroja errores de autenticacion
ni de permisos.

---

## Resolucion de problemas comunes

### El navegador rechaza la conexion al autenticar con Google Drive

Este es el problema mas frecuente con OAuth en DVC. Lo que ocurre es:

1. `dvc push` arranca un servidor local en `localhost:8080`
2. Abre el navegador a la pagina de consent de Google
3. Al aprobar los permisos, Google redirige a `localhost:8080`
4. Si esa conexion falla, el navegador muestra "Esta pagina no funciona" o "Conexion rechazada"

**Opcion A — URL manual (lo mas rapido):**

Cuando el navegador falle, busca en la terminal un mensaje como:
```
Please open the following URL in your browser:
https://accounts.google.com/o/oauth2/auth?...
```
Abre ese link, aprueba los permisos en Google, y cuando el navegador
muestre "Esta pagina no funciona" en `localhost:8080`, copia la URL
completa de la barra del navegador (incluye el parametro `?code=...`)
y pegala en la terminal cuando te la solicite.

**Opcion B — Service Account (la mas confiable):**

Evita el navegador completamente. Usa una cuenta de servicio de Google Cloud:

1. Ve a [console.cloud.google.com](https://console.cloud.google.com)
2. Crea un proyecto nuevo (o usa uno existente)
3. En **APIs y Servicios** → **Biblioteca**, habilita la **Google Drive API**
4. En **APIs y Servicios** → **Credenciales**, crea una **Cuenta de Servicio**
5. Descarga el archivo JSON de la clave de la cuenta de servicio
6. En Google Drive, **comparte el folder `mibici-dvc-remote`** con el email
   de la cuenta de servicio (lo encuentras en el JSON como `"client_email"`)

Luego configura DVC:
```bash
dvc remote modify gdrive gdrive_use_service_account true
dvc remote modify gdrive gdrive_service_account_json_file_path "ruta/al/credenciales.json"

# Verifica que funcione
dvc push
```

La ventaja: no requiere navegador ni interaccion manual. Funciona
en cualquier entorno, incluyendo Windows PowerShell, CI/CD y servidores remotos.

> **Nota de seguridad:** el archivo JSON de la cuenta de servicio contiene
> credenciales privadas. Nunca lo incluyas en el repositorio de git.
> Agrega la ruta al `.gitignore` si es necesario.

---

### "No such remote 'gdrive'"
Olvidaste configurar el remote. Ejecuta el Paso 2.

### Error de autenticacion de Google (token expirado)
El token expiro. Elimina la cache de credenciales:
```bash
rm -rf ~/.cache/pydrive2fs/   # Linux/Mac
# Windows: elimina %LOCALAPPDATA%\pydrive2fs\
```
Luego ejecuta `dvc push` o `dvc pull` de nuevo.

### "No module named 'dvc_gdrive'"
Falta el extra de Google Drive. Instala:
```bash
pip install dvc-gdrive==3.0.1
```
Actualiza `requirements.txt` con esa version.

### "dvc: command not found"
DVC no esta instalado en este entorno. Activa el entorno virtual:
```bash
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```
O instalalo directamente: `pip install dvc==3.67.1 dvc-gdrive==3.0.1`.

### "Data and pipelines are up to date" pero los archivos no existen
Ejecuta `dvc checkout` para restaurar los archivos desde el cache local,
o `dvc pull` para descargarlos desde el remote.

---

## Archivos creados por DVC (que deben estar en git)

| Archivo | Descripcion |
|---|---|
| `dvc.yaml` | Definicion del pipeline (stages, deps, outs, params, metrics) |
| `dvc.lock` | Estado actual de los artefactos (hashes); se actualiza con `dvc repro` |
| `.dvc/config` | Configuracion del remote |
| `.dvc/.gitignore` | Excluye cache y tmp del control de versiones |
| `.dvcignore` | Patrones de archivos que DVC debe ignorar |

**Nunca** commits los archivos dentro de `.dvc/cache/`; esos son los
datos binarios que DVC maneja internamente.
