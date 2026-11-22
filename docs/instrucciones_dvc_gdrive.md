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

Si los artefactos YA existen localmente (el pipeline se ejecuto con make):

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
make download

# Luego reproduce el pipeline completo via DVC
dvc repro

# Sube los artefactos al remote
dvc push
```

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
         *
   +----------+
   | features |
   +----------+
    *        *
  **          *
 *             **
+-------+       *
| train |      **
+-------+     *
    *        *
     **    **
       *  *
   +---------+
   | predict |
   +---------+
```

Criterio cumplido si el DAG muestra las 4 etapas en orden correcto.

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

### "No such remote 'gdrive'"
Olvidaste configurar el remote. Ejecuta el Paso 2.

### Error de autenticacion de Google
El token expiro. Elimina la cache de credenciales:
```bash
rm -rf ~/.cache/pydrive2fs/   # Linux/Mac
# Windows: elimina el folder AppData\Local\pydrive2fs
```
Luego ejecuta `dvc push` o `dvc pull` de nuevo; abrira el navegador.

### "dvc: command not found"
DVC no esta instalado en este entorno. Activa el entorno virtual:
```bash
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```
O instalalo directamente: `pip install dvc==3.67.1`.

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
