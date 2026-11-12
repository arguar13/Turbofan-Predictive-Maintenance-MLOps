# Predictive Maintenance MLOps

**Plataforma MLOps para clasificación de Vida Útil Remanente (RUL) en motores turbofan**

*Read this in other languages: [English](README.md)*

[![Python](https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white)](api/pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-ConvTransformer-EE4C2C?logo=pytorch&logoColor=white)](core_ml/src/train.py)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking%20%2B%20Registry-0194E2?logo=mlflow&logoColor=white)](core_ml/src/train.py)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Kustomize-326CE5?logo=kubernetes&logoColor=white)](kubernetes/base)
[![Terraform](https://img.shields.io/badge/IaC-Terraform-7B42BC?logo=terraform&logoColor=white)](terraform)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitLab%20CI-FC6D26?logo=gitlab&logoColor=white)](.gitlab-ci.yml)
[![CI](https://github.com/arguar13/Turbofan-Predictive-Maintenance-MLOps/actions/workflows/ci.yml/badge.svg?branch=dev)](.github/workflows/ci.yml)

## Qué hace este proyecto

Predice la degradación de motores turbofan usando el dataset C-MAPSS de la NASA. El problema se plantea como una clasificación de 3 clases: **Healthy** (saludable), **Alert** (alerta), **Critical** (crítico), servida por un **ConvTransformer** (una capa `Conv1d` seguida de un `TransformerEncoder`) entrenado en PyTorch.

Lo interesante no es solo el modelo, sino todo lo que lo rodea para que una predicción sea confiable y reproducible. Los logs crudos de sensores pasan por una preparación de datos versionada y validada contra un esquema; cada corrida de entrenamiento queda atada al código, los datos y la imagen de contenedor exactos que la produjeron; un gate de promoción evita que un modelo que no detecta bien la clase "a punto de fallar" llegue a producción; y todo esto se despliega a un cluster de Kubernetes real mediante infraestructura como código, no a mano.

Cada capa resuelve un problema concreto de este pipeline en particular: DVC fija *qué* dataset produjo un modelo, los contratos de datos con Pandera/Pydantic detectan telemetría malformada antes de desperdiciar una corrida de entrenamiento, el Model Registry de MLflow es la única fuente de verdad sobre "qué modelo está sirviendo ahora mismo", y la capa de FastAPI confía en ese registro en vez de en un `.pkl` que alguien copió a mano.

## Arquitectura

```mermaid
graph TD
    subgraph Infra["Infraestructura (Terraform)"]
        TF[Terraform] --> VPC[VPC, 2 AZ]
        TF --> EKS[EKS: 1 node group]
        TF --> RDS[(RDS PostgreSQL)]
        TF --> S3[(S3: datos + artefactos)]
        TF --> ECR[(ECR: imagen API)]
    end

    subgraph CI["GitLab CI"]
        LINT[lint_test] --> BUILD[build_push]
        BUILD --> DEPLOY[deploy]
        TRAINJOB["train (manual)"]
    end

    DVC[(Dataset C-MAPSS, DVC)] --> TRAINJOB
    TRAINJOB -->|entrena ConvTransformer + quality gate| MLFLOW[(MLflow: Tracking + Registry)]
    MLFLOW --> RDS
    MLFLOW --> S3
    BUILD -->|push imagen| ECR
    ECR --> DEPLOY
    DEPLOY -->|kubectl apply -k| API[API FastAPI]
    DEPLOY --> MLFLOWSVC[MLflow Server]
    API -->|carga modelo champion| MLFLOWSVC
    MLFLOWSVC -.-> MLFLOW
```

**Flujo, en palabras:**

1. Los datos crudos de C-MAPSS (versionados con DVC) se transforman con `prepare_training_data.py` en ventanas deslizantes de 30 timesteps sin escalar (`engine_features.parquet` + `feature_names.json`), a partir de una muestra de motores completos repartida por igual entre FD001–FD004.
2. `train.py` reparte los motores en train/val/test, ajusta el `StandardScaler` solo con los motores de train, entrena el ConvTransformer y registra la corrida en MLflow (modelo + scaler + métricas + tags de trazabilidad). Solo se registra en el Model Registry y pasa al alias `champion` si supera el quality gate y no es peor que el champion actual.
3. La API FastAPI carga el `champion` y su scaler desde MLflow (reintentando hasta que exista uno), y sirve `/predict`.
4. GitLab CI: `lint_test` (formato + ruff + pytest) en cada push; `build_push` (build + push a ECR) y un `deploy` manual (`kubectl apply -k`) solo en la rama por defecto. `train` es un job manual aparte, porque una corrida completa de entrenamiento puede tardar de minutos a horas y no debería bloquear cada push. GitHub Actions corre los mismos checks de calidad más un build de Docker, sin acceso a AWS.

**Algunas decisiones de diseño que vale la pena señalar:**

- **La inferencia es sincrónica, bajo demanda.** Un cliente envía una ventana de lecturas a `/predict` y recibe una clasificación en la misma petición; no hay ningún broker de mensajes ni capa de streaming en el camino, porque nada en este sistema necesita reaccionar a un flujo continuo más rápido de lo que permite un ciclo request/response.
- **Las features se calculan una sola vez, offline.** `prepare_training_data.py` transforma la telemetría cruda en la forma de tensor exacta que consume el modelo, y `train.py` lee ese parquet directamente. Como las predicciones se sirven a partir de una ventana que provee quien llama, y no de valores de feature en vivo constantemente actualizados, no hace falta un store online aparte que mantener sincronizado.
- **El despliegue es un único paso auditable.** El stage `deploy` de CI se autentica contra AWS vía OIDC y corre `kubectl apply -k` contra el overlay ya renderizado. Desde un commit mergeado hasta un pod corriendo hay exactamente un sistema involucrado, lo que hace fácil de trazar el camino cuando algo necesita depurarse.
- **Los secretos son Secrets nativos de Kubernetes.** Las credenciales de RDS, la URI de MLflow y la API key llegan a los pods vía `envFrom: secretRef` El Secret se crea una vez por cluster con `kubectl create secret` y deliberadamente **no** forma parte de los recursos de Kustomize (`kubernetes/base/secret.yaml` es solo una plantilla de las claves esperadas), así `kubectl apply -k` nunca puede sobrescribir credenciales reales con placeholders.
- **El acceso a S3 viaja sobre el rol IAM del node group.** Todos los pods del cluster heredan la misma política de lectura/escritura acotada para los buckets de DVC y de MLflow, así que hay una sola política que razonar en vez de una por servicio.

## Stack tecnológico

| Categoría | Herramientas |
| --- | --- |
| Machine Learning | PyTorch (CPU-only), ConvTransformer |
| Tracking / Model Registry | MLflow |
| Datos y contratos | DVC (remoto S3), Pandera, Pydantic |
| API de serving | FastAPI, Uvicorn |
| Contenedores / Orquestación | Docker, Docker Compose, Kubernetes (EKS), Kustomize |
| Infraestructura como Código | Terraform (VPC, EKS, RDS, S3, ECR, IAM/OIDC) |
| CI/CD | GitLab CI (autenticación AWS vía OIDC), GitHub Actions (solo calidad) |
| Calidad de código / observabilidad | Ruff, mypy (opcional), pytest, pre-commit, structlog |

## El pipeline de MLOps

**Datos.** Cada etapa valida su propia salida contra un esquema explícito (`core_ml/src/data_contracts.py`, con Pandera) antes de avanzar a la siguiente, más costosa. Un dataset malformado o una lectura `NaN` se rechaza de inmediato. La RUL se deriva por motor como `max(ciclo) - ciclo` y se agrupa en 3 clases: `RUL > 60` → Healthy, `30 < RUL <= 60` → Alert, `RUL <= 30` → Critical.

**El modelo.** `ConvTransformer` (`core_ml/src/train.py`) pasa una capa `Conv1d` sobre la ventana de 30 timesteps para extraer patrones locales de sensores, agrega codificación posicional sinusoidal, y alimenta un `TransformerEncoder` de 2 capas que modela dependencias de largo alcance. La convolución captura tendencias cortas entre sensores correlacionados con pocos parámetros, y la atención relaciona cualquier par de timesteps de la ventana sin el cuello de botella secuencial de una RNN. Es deliberadamente pequeño (`d_model=64`, 2 capas) para que el dataset completo entrene en CPU.

**Quality gate.** Los motores se reparten en tres grupos, nunca fila a fila (las ventanas se solapan y un split IID filtraría casi-duplicados): **train** (60%) ajusta los pesos, **val** (20%) elige el mejor checkpoint y decide el early stopping, y **test** (20%) se usa una sola vez, sobre el checkpoint restaurado, para decidir el gate. Evaluar el gate sobre val sería optimista, porque val ya eligió el epoch. La promoción al alias `champion` exige superar **dos** umbrales en `config/config.yaml`: `f2_weighted_threshold` **y** `critical_recall_threshold` (un piso duro sobre el recall de la clase Critical, para que un modelo no compense fallar justo ahí con buen desempeño en las otras clases). Pasar el gate no alcanza para reemplazar al champion: si el champion actual se evaluó sobre exactamente los mismos datos y el mismo split (misma tag `training_data_fingerprint`), el retador además tiene que igualar o superar su F2. Solo los runs promovidos obtienen una versión en el Model Registry; los rechazados quedan como runs de MLflow, auditables pero nunca servibles. Cada run registra además las métricas que exponen las debilidades en vez de ocultarlas: F1 macro, precisión/recall/F1 por clase, una línea base de clase mayoritaria y la matriz de confusión de test (`evaluation/test_report.json`).

**Resultados.** Dataset completo (709 motores), evaluado una sola vez sobre 142 motores de test (27.212 ventanas) que nunca se usaron para entrenar ni para elegir el modelo:

| Métrica | Modelo | Línea base "siempre Healthy" |
| --- | --- | --- |
| Accuracy | 0,894 | 0,682 |
| F2 ponderado (gate ≥ 0,75) | 0,893 | 0,623 |
| Recall de Critical (gate ≥ 0,75) | 0,884 | 0,000 |
| F1 macro | 0,831 | 0,270 |
| Recall de Alert (clase más débil) | 0,626 | 0,000 |

El modelo supera ambos umbrales del gate; la banda Alert (31–60 ciclos restantes) concentra la mayoría de los errores, por eso el F1 macro se reporta junto a las métricas ponderadas.

**Dataset toy.** `core_ml/data_toy/` (1116 filas, versionado con DVC, o regenerado en local con `make build-toy-dataset`) permite correr todo el pipeline en segundos, sin GPU; es lo que corre `make smoke-test`. CI no necesita DVC: `core_ml/tests/test_pipeline_e2e.py` ejecuta prepare → train → MLflow → Registry de punta a punta sobre un dataset sintético con el formato exacto de C-MAPSS.

**Serving.** `api/main.py` carga el modelo `champion` y su `StandardScaler` desde la misma versión de MLflow. Si MLflow todavía no responde o no hay champion, reintenta (cada 30 s) en vez de quedarse en `503` hasta un reinicio. La forma esperada y el orden de columnas salen de los artefactos servidos (scaler + signature del modelo), nunca de una constante en `config.yaml`, así el contrato de la API sigue automáticamente a cualquier modelo promovido.

## Referencia de la API

| Endpoint | Método | Descripción |
| --- | --- | --- |
| `/health` | `GET` | `200` con `{"status": "ok", "model_version": "<n>", "window_size": <n>, "num_features": <n>, "feature_names": [...]}` si hay un modelo `champion` cargado; `503` si no |
| `/predict` | `POST` | Requiere header `X-API-Key`. Body: `{"engine_id": "...", "readings": [[...], ...]}` (una fila por timestep, columnas en el orden de `feature_names`, valores crudos sin escalar). Devuelve `{"engine_id": "...", "prediction": "Healthy"\|"Alert"\|"Critical", "probabilities": {...}, "model_version": "<n>"}` |

`/health` se usa como readiness probe. El liveness probe usa un chequeo TCP simple (no `/health`): un pod sano que todavía no tiene modelo `champion` es un estado de negocio, no un proceso muerto, y no debería reiniciarse en bucle por eso.

## Estructura del repositorio

```text
.
├── api/                    # Microservicio FastAPI (proyecto Poetry propio)
│   ├── main.py               # Carga models:/<name>@champion y sirve /predict
│   └── tests/                 # Tests unitarios (TestClient/mocks)
├── core_ml/                # Datos + entrenamiento (proyecto Poetry propio)
│   ├── src/
│   │   ├── data_contracts.py    # Contratos Pandera (fail fast)
│   │   ├── data_processing.py   # Limpieza + ventanas deslizantes + scaler solo de train
│   │   ├── data_paths.py        # Ubicación de crudos (DVC) y derivados (prepared/)
│   │   ├── prepare_training_data.py  # Genera engine_features.parquet
│   │   └── train.py             # ConvTransformer + quality gate + MLflow
│   ├── data/ · data_toy/        # Datasets crudos (versionados con DVC, gitignored)
│   ├── prepared/                # Ventanas derivadas por dataset (gitignored, regenerables)
│   └── tests/                    # Tests unitarios
├── config/config.yaml       # Configuración global (validada por Pydantic)
├── kubernetes/
│   ├── base/                # Namespace, ServiceAccount, ConfigMap, plantilla de Secret,
│   │                          #   Deployments de api/mlflow, HPA
│   └── overlays/production/ # Tag de imagen (kustomize edit set image)
├── terraform/                # VPC, EKS, RDS, S3, ECR, IAM (OIDC de CI)
│   └── bootstrap/              # Una sola vez: backend S3+DynamoDB del estado
├── localstack/init-aws.sh   # Bootstrap de LocalStack (S3 simulado en local)
├── .gitlab-ci.yml           # lint_test → build_push → deploy → train (manual)
├── .github/workflows/ci.yml # GitHub: formato + lint + tests + docker build (sin AWS)
├── Makefile                 # Interfaz única de comandos (local + CI)
├── docker-compose.yml       # Stack local: Postgres, MLflow, API, LocalStack
└── Dockerfile                # Imagen de la API
```

## Cómo correrlo en local

### 1. Prerrequisitos

Python 3.12, Poetry 2.4.1, Docker (con Compose v2), `make`, `pre-commit`.

### 2. Instalar dependencias y hooks

```bash
pip install "poetry==2.4.1" pre-commit
make install              # api/ y core_ml/, con dependencias de dev
make precommit-install    # habilita los git hooks
```

### 3. Calidad de código y tests

```bash
make lint       # ruff check
make format     # ruff format (autocorrige)
make typecheck  # mypy (opcional, no bloquea CI)
make test       # pytest en api/ y core_ml/
make ci         # lo mismo que corre el job lint_test de GitLab CI
```

### 4. Validar el pipeline con el dataset toy (segundos, sin GPU)

```bash
# primero copia los archivos de C-MAPSS en core_ml/data/ (ver "Cómo obtener los datos")
make build-toy-dataset   # genera data_toy/ (5 motores) desde core_ml/data/
make smoke-test          # contratos + entrenamiento + MLflow + quality gate
```

`make smoke-test` es autónomo: si no hay `MLFLOW_TRACKING_URI` exportado, usa un tracking store SQLite efímero.

### 5. Levantar el stack completo con Docker Compose

```bash
make compose-up   # Postgres, MLflow, API, LocalStack
make compose-ps   # healthcheck de cada servicio
```

| Servicio | URL en el host |
| --- | --- |
| API | `http://localhost:8001` (`/health`, `/predict`, `/docs`) |
| MLflow UI | `http://localhost:5001` |
| LocalStack | `http://localhost:4567` |
| Postgres | `localhost:5433` |

`/health` devuelve `503` hasta que un modelo tenga el alias `champion` en este MLflow local; es el estado inicial esperado. Al entrenar contra él, la API toma el champion sola en ~30 s:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5001
make train-toy              # segundos: prueba la integración (o `make train` para un modelo real)
docker compose restart api  # solo hace falta para servir un champion NUEVO más adelante
```

Un champion entrenado con el dataset toy (5 motores, 1 de ellos en validación) solo demuestra que la integración funciona; sus métricas no son representativas.

Los puertos del host están corridos (8001, 5001, 4567, 5433 en vez de los defaults) para que este stack pueda convivir con otros proyectos en la misma máquina sin pisarse; los puertos internos de los contenedores no cambian.

## Cómo obtener los datos

El modelo se entrena con el dataset público C-MAPSS de la NASA (A. Saxena y K. Goebel, 2008, *Turbofan Engine Degradation Simulation Data Set*, NASA Prognostics Data Repository, NASA Ames Research Center). Descarga la entrada **6. Turbofan Engine Degradation Simulation** del [NASA Prognostics Center of Excellence Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/), descomprímela y copia `train_FD001.txt` … `train_FD004.txt` en `core_ml/data/` (el pipeline solo lee los cuatro archivos de entrenamiento).

`make dvc-pull` hace lo mismo desde el remoto DVC del proyecto, que vive en el bucket S3 privado del autor, así que solo funciona con acceso a esa cuenta de AWS.

## Cómo entrenar el modelo

```bash
make train         # prepara features + entrena (25 epochs, patience 5) + quality gate (~30 min en CPU)
```

Por debajo, `make train` encadena las dos etapas del pipeline:

1. **`prepare_training_data.py`** lee los `.txt` crudos de C-MAPSS, muestrea motores completos repartidos por igual entre FD001–FD004 (los 709 motores por defecto; `--max-engines N` toma una muestra estratificada más rápida), los limpia y los transforma en ventanas (`data_processing.py`), y escribe las ventanas sin escalar (`engine_features.parquet`) y el orden de sus columnas (`feature_names.json`) en `core_ml/prepared/<dataset>/`. Ese directorio está ignorado por git y fuera de lo que versiona DVC, así que `data.dvc` describe exactamente los archivos crudos: escribir artefactos derivados junto a ellos era lo que dejaba el puntero desactualizado.
2. **`train.py`** carga ese parquet, lo valida contra su contrato de datos, reparte los motores en train/val/test, ajusta el scaler solo con los motores de train, entrena el `ConvTransformer` y registra la corrida en MLflow: modelo, scaler, métricas, y una tupla de trazabilidad con `commit de git (+ marca git_dirty) + hash de datos de DVC + hiperparámetros + ID de run de MLflow + tag de imagen de contenedor`. Solo registra el modelo y le mueve el alias `champion` si supera el quality gate y no es peor que un champion comparable.

`train.py` es reproducible (`torch.manual_seed(42)`, split train/val/test seedeado y agrupado por motor): dado un run de MLflow, siempre se puede reconstruir con qué commit, qué versión de datos y qué imagen se generó, vía las tags de trazabilidad del run. En GitLab CI, el job `train` es manual (no corre en cada push) porque entrenar contra el dataset completo puede tardar minutos u horas.

## Cómo servir predicciones

Una vez que un modelo fue promovido a `champion`, `api/main.py` lo carga junto con su scaler desde el Model Registry de MLflow y expone `/predict`. `GET /health` devuelve el `window_size` esperado y el orden `feature_names` de cada fila. Una petición se ve así:

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: local-dev-only-key-change-me" \
  -d '{"engine_id": "FD001_23", "readings": [[...30 filas de valores de sensores...]]}'
```

La respuesta trae la clase predicha (`Healthy` | `Alert` | `Critical`), la probabilidad de cada clase y la versión del modelo que la produjo, de modo que quien llama siempre puede saber qué modelo registrado respondió una petición dada. Las peticiones con una forma de ventana incorrecta, valores de sensor no finitos, o un `engine_id` ausente/vacío se rechazan con `422` antes de llegar al modelo (`api/schemas.py` y el chequeo de forma en `api/main.py`); las peticiones sin un `X-API-Key` válido se rechazan con `401`.

## Cómo desplegar en AWS

> **Desplegado en una cuenta real de AWS.** Este stack se desplegó de punta a punta en AWS: Terraform creó la infraestructura, la imagen se subió a ECR y el servicio corrió en EKS, con MLflow respaldado por RDS y S3. Después se eliminó con `terraform destroy` para no acumular costos (EKS, RDS y el NAT Gateway se cobran por hora), así que no hay un endpoint activo. Los pasos de abajo son los que se usaron en ese despliegue.

Esto asume una cuenta de AWS propia y cuesta dinero real (EKS + RDS + NAT Gateway, aunque sea con instancias chicas). Pasos, en orden:

```bash
# 1. Una sola vez: backend remoto del estado de Terraform
cd terraform/bootstrap && terraform init && terraform apply
terraform output -raw backend_config > ../backend.hcl   # ignorado por git: lleva el ID de la cuenta

# 2. Aprovisionar la infraestructura y apuntar kubectl al cluster nuevo
#    (terraform.tfvars, también ignorado por git, define tu IP pública: ver terraform.tfvars.example)
cd terraform && terraform init -backend-config=backend.hcl -migrate-state && terraform apply
aws eks update-kubeconfig --name predictive-maintenance-mlops --region us-east-1

# 3. Crear el namespace y el Secret de Kubernetes con las credenciales reales
#    (el Secret no forma parte de los recursos de Kustomize, así los despliegues nunca lo pisan)
kubectl create namespace mlops-env --dry-run=client -o yaml | kubectl apply -f -
#    (terraform output -raw db_password / db_username; ver kubernetes/base/secret.yaml)
kubectl create secret generic mlops-secrets -n mlops-env \
  --from-literal=POSTGRES_USER=<...> --from-literal=POSTGRES_PASSWORD=<...> \
  --from-literal=POSTGRES_DB=mlflow_db --from-literal=API_KEY="$(openssl rand -hex 32)" \
  --dry-run=client -o yaml | kubectl apply -f -

# 4. Completar los valores propios de la cuenta, que el repo público deja como marcadores:
#    REDACTED-RDS-ENDPOINT en kubernetes/base/configmap.yaml (terraform output -raw rds_endpoint)
#    REDACTED_ACCOUNT_ID en kubernetes/base/mlflow.yaml (terraform output -raw s3_bucket_name)
# 5. Build + push de la imagen de la API, y despliegue
make docker-build docker-push deploy AWS_ACCOUNT_ID=<tu-account-id> IMAGE_TAG=$(git rev-parse --short HEAD)
```

En GitLab CI, `build_push` hace esto automáticamente en los push a la rama por defecto, y `deploy` es un botón manual de ese pipeline (ambos se autentican contra AWS vía OIDC, sin credenciales estáticas; ver `terraform/iam.tf`). Las ramas de feature solo corren `lint_test`. `make k8s-build` renderiza el overlay para inspección antes de aplicar; `make k8s-diff` muestra el diff contra el cluster actual.

Terraform provisiona una VPC en 2 zonas de disponibilidad, un cluster EKS de un node group, una instancia RDS PostgreSQL (el backend store de MLflow), un bucket S3 (datos de DVC + artefactos de MLflow), un repositorio ECR para la imagen de la API, y los roles IAM / proveedores OIDC que CI necesita para autenticarse sin credenciales de larga duración.

## Testing y calidad

| Aspecto | Herramienta | Comando |
| --- | --- | --- |
| Lint | [Ruff](https://docs.astral.sh/ruff/) | `make lint` |
| Formato | Ruff format | `make format-check` |
| Tipado estático (opcional) | [mypy](https://mypy-lang.org/) | `make typecheck` |
| Tests unitarios | [pytest](https://docs.pytest.org/) | `make test` |
| Higiene de archivos/YAML | pre-commit hooks | `make precommit-run` |

`make ci` corre lo mismo que el job `lint_test` de `.gitlab-ci.yml` y el job `lint_test` de `.github/workflows/ci.yml`: si pasa en local, el pipeline de CI también debería pasar. Tanto `api/` como `core_ml/` son proyectos Poetry independientes, cada uno con su propio `pyproject.toml`, lockfile y suite de tests, y pre-commit corre los mismos comandos de ruff/mypy/pytest contra el proyecto cuyos archivos cambiaron en un commit.

**Autor:** Armando Guarnera, [github.com/arguar13](https://github.com/arguar13)
