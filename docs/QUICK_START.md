# Guía de Inicio Rápido

**Sistema de Recuperación de Información — Tecnología y Software**

---

## Requisitos Previos

```bash
# Verificar versión de Python (se requiere 3.12)
python --version

# Verificar pip
pip --version
```

---

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/D4R102004/Information-Retrieval-System-Project.git
cd Information-Retrieval-System-Project

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Instalar dependencias
pip install -e ".[dev]"
```

---

## Instalar y Configurar Ollama

El módulo RAG requiere Ollama para ejecutar el LLM localmente.

```bash
# Descargar e instalar desde https://ollama.ai

# Descargar el modelo
ollama pull llama3.2:latest

# Iniciar el servicio (en una terminal separada)
ollama serve
```

---

## Despliegue con Docker

Alternativa recomendada para reproducibilidad sin configurar dependencias manualmente:

```bash
# Construir las imágenes
docker compose build

# Ejecutar el rastreador para recopilar documentos
docker compose run crawler

# Ejecutar el sistema completo
docker compose run demo
```

---

## Interfaz Visual (Gradio)

```bash
# Activar el entorno virtual
source .venv/bin/activate

# Iniciar la interfaz
PYTHONPATH=src python -m ui.app
```

Abrir en el navegador: http://127.0.0.1:7860

### Pestañas disponibles

**Búsqueda:** Ingresar una consulta en lenguaje natural y presionar *Search*. Se muestran los documentos recuperados con fuente, puntuación y fragmento, seguidos de la respuesta RAG con citas.

**Configuración:** Ajustar parámetros antes de buscar y presionar *Save Query Settings* para aplicarlos.

- *Max Local Results* (por defecto 5): documentos del índice local
- *Max Web Results* (por defecto 5): documentos del fallback web
- *Enable Web Search*: activar DuckDuckGo cuando los resultados locales sean insuficientes
- *Ollama Model*, *Temperature*, *Max Tokens*, *Max Citations*

**Evaluación:** Cargar consultas de prueba, ejecutar la evaluación y visualizar métricas (MAP, MRR, NDCG@5, P@5, R@10).

**Estado del Sistema:** Ver el estado de la base de datos, los rastreadores y el servicio LLM.

---

## Uso por Línea de Comandos

```bash
# Consulta única
PYTHONPATH=src python src/main.py --query "What is machine learning?"

# Cargar datos desde los rastreadores
PYTHONPATH=src python src/main.py --load-data

# Ver estado de la base de datos
PYTHONPATH=src python src/main.py --status

# Modo interactivo
PYTHONPATH=src python src/main.py --interactive

# Ejecutar evaluación
PYTHONPATH=src python src/main.py --evaluate
```

---

## Uso desde Python

```python
import sys
sys.path.insert(0, "src")

from main_orchestator import MainOrchestator

orchestrator = MainOrchestator()

# Consulta completa (recuperación + RAG)
response = orchestrator.query("What is Docker containerization?")
print(response.answer)
print([c.title for c in response.citations])

# Solo recuperación, sin generación RAG
result = orchestrator.retrieve_documents(
    question="machine learning frameworks",
    max_local_results=10,
    enable_web_search=True,
)
documents = result["documents"]

# Estado del sistema
status = orchestrator.get_status()
print(f"Documentos indexados: {status['database']['indexed_documents']}")

# Cargar datos
result = orchestrator.load_documents_from_crawlers(max_articles=500)
print(f"Indexados: {result['indexed_documents']} documentos")
```

---

## Ejecutar Tests

```bash
# Tests del rastreador y búsqueda web (sin dependencias externas)
PYTHONPATH=src python -m pytest tests/sri/crawler/ tests/sri/web_search/ -v

# Tests del módulo RAG (sin LLM activo)
PYTHONPATH=src python -m pytest tests/sri/rag/ \
  --ignore=tests/sri/rag/test_pipeline_integration.py \
  --ignore=tests/sri/rag/test_rag_module_with_llm.py -v

# Tests del módulo de recomendación
PYTHONPATH=src python -m pytest tests/test_recommender.py tests/test_user_history.py -v
```

---

## Calidad de Código

```bash
# Formatear código
ruff format src/

# Verificar estilo
ruff check src/

# Ejecutar todos los checks
make check
```

---

## Solución de Problemas Comunes

**Error: "Ollama connection failed"**
```bash
# Iniciar el servicio en otra terminal
ollama serve
# Verificar que el modelo está descargado
ollama list
```

**Error: "Database is empty"**
```bash
PYTHONPATH=src python src/main.py --load-data
```

**Error: "ChromaDB not installed"**

El sistema usa automáticamente el backend local como alternativa. Para instalar ChromaDB explícitamente:
```bash
pip install chromadb
```

**Error: "Import failed" o "ModuleNotFoundError"**

Verificar que el entorno virtual está activo y que `PYTHONPATH=src` está configurado:
```bash
source .venv/bin/activate
PYTHONPATH=src python src/main.py --status
```

---

## Flujo de Desarrollo

```bash
# 1. Crear rama para la tarea
git checkout -b feat/issue-N-descripcion

# 2. Realizar cambios y ejecutar tests
PYTHONPATH=src python -m pytest tests/sri/crawler/ tests/sri/web_search/ -v

# 3. Formatear y verificar estilo
make format && make lint

# 4. Confirmar cambios con formato convencional
git add .
git commit -m "feat(modulo): descripción del cambio

Refs #N"

# 5. Abrir Pull Request para revisión
git push origin feat/issue-N-descripcion
gh pr create --base main
```

---

Para documentación técnica detallada ver:

- `docs/paper/paper.pdf` — artículo técnico en formato LNCS
- `docs/BACKEND_ARCHITECTURE.md` — análisis de arquitectura
- `docs/RECOMMENDATION_MODULE.md` — módulo de recomendación
- `README.md` — documentación completa del proyecto
