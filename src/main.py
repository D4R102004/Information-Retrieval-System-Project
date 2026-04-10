"""
demo_and_test.py
=================
Script de demostración y prueba de integración del SRI para
el dominio Tecnología y Software.

Ejecutar:
    python demo_and_test.py
"""

import sys
import os
import json

# Asegurar que los módulos sean accesibles
sys.path.insert(0, os.path.dirname(__file__))

from sri.pipeline import SRIPipeline
from evaluation.evaluation import Evaluator


# ---------------------------------------------------------------------------
# Corpus de ejemplo — dominio Tecnología y Software
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Python para Machine Learning: Guía Completa 2024",
        "content": (
            "Python se ha convertido en el lenguaje de referencia para machine learning "
            "e inteligencia artificial. Frameworks como TensorFlow, PyTorch y scikit-learn "
            "permiten construir modelos de deep learning y aprendizaje automático de forma "
            "sencilla. Este artículo explora las mejores prácticas, herramientas y librerías "
            "para proyectos de IA con Python, incluyendo gestión de datos con pandas, "
            "visualización con matplotlib y despliegue con FastAPI."
        ),
        "url": "https://tech.example.com/python-ml-guide",
        "tags": ["python", "machine learning", "tensorflow", "pytorch", "ia"],
        "type": "tutorial",
        "date": "2024-03-15",
        "popularity": 8500,
    },
    {
        "id": "doc_002",
        "title": "Docker y Kubernetes: Contenedores en Producción",
        "content": (
            "Docker y Kubernetes son las tecnologías clave para el despliegue de aplicaciones "
            "en contenedores. Docker permite empaquetar aplicaciones con sus dependencias, "
            "mientras Kubernetes orquesta múltiples contenedores a escala. Aprende a crear "
            "Dockerfiles optimizados, configurar clusters de Kubernetes, implementar estrategias "
            "de rolling deployment y gestionar microservicios con Helm charts."
        ),
        "url": "https://tech.example.com/docker-kubernetes",
        "tags": ["docker", "kubernetes", "devops", "contenedores", "microservicios"],
        "type": "article",
        "date": "2024-02-20",
        "popularity": 6200,
    },
    {
        "id": "doc_003",
        "title": "React vs Vue vs Angular: Comparativa 2024",
        "content": (
            "Los tres grandes frameworks de JavaScript frontend siguen dominando el desarrollo "
            "web. React con su arquitectura de componentes y virtual DOM, Vue con su curva de "
            "aprendizaje suave y Vue 3 Composition API, y Angular con su solución empresarial "
            "completa. Analizamos rendimiento, ecosistema, curva de aprendizaje y casos de uso "
            "para ayudarte a elegir el framework correcto para tu proyecto."
        ),
        "url": "https://tech.example.com/react-vue-angular-2024",
        "tags": ["react", "vue", "angular", "javascript", "frontend", "web"],
        "type": "article",
        "date": "2024-01-10",
        "popularity": 12000,
    },
    {
        "id": "doc_004",
        "title": "Introducción a los Large Language Models (LLMs)",
        "content": (
            "Los modelos de lenguaje grandes como GPT-4, Claude, Llama y Gemini han "
            "revolucionado el procesamiento del lenguaje natural. Basados en arquitecturas "
            "Transformer con mecanismos de atención, estos modelos son entrenados en "
            "enormes corpus de texto. Exploramos cómo funcionan internamente, técnicas de "
            "fine-tuning, prompt engineering, y aplicaciones en generación de código, "
            "resumen, traducción y sistemas RAG."
        ),
        "url": "https://tech.example.com/llm-introduction",
        "tags": ["llm", "gpt", "transformer", "nlp", "ia", "claude", "prompt"],
        "type": "article",
        "date": "2024-04-01",
        "popularity": 15000,
    },
    {
        "id": "doc_005",
        "title": "Git Avanzado: Workflows y Mejores Prácticas",
        "content": (
            "Dominar Git va más allá de commit y push. Este artículo cubre estrategias "
            "avanzadas como GitFlow, trunk-based development, cherry-pick, rebase interactivo, "
            "git bisect para debugging, submódulos, y configuración de hooks para CI/CD. "
            "También abordamos manejo de conflictos en equipos grandes y estrategias de "
            "branching para proyectos open source."
        ),
        "url": "https://tech.example.com/git-advanced",
        "tags": ["git", "devops", "version control", "cicd", "open source"],
        "type": "tutorial",
        "date": "2023-11-05",
        "popularity": 4300,
    },
    {
        "id": "doc_006",
        "title": "Bases de Datos Vectoriales: ChromaDB, Pinecone, Weaviate",
        "content": (
            "Las bases de datos vectoriales son fundamentales para aplicaciones de IA modernas, "
            "especialmente sistemas RAG y búsqueda semántica. ChromaDB ofrece una solución "
            "embebida ideal para desarrollo, Pinecone escala a millones de vectores en la nube, "
            "y Weaviate combina búsqueda vectorial con capacidades de grafo. Comparamos "
            "rendimiento, API, costos y casos de uso de cada plataforma."
        ),
        "url": "https://tech.example.com/vector-databases",
        "tags": ["chromadb", "pinecone", "weaviate", "rag", "embeddings", "ia"],
        "type": "article",
        "date": "2024-03-28",
        "popularity": 7800,
    },
    {
        "id": "doc_007",
        "title": "Seguridad en APIs REST: OAuth 2.0 y JWT",
        "content": (
            "Proteger APIs REST es crítico en cualquier aplicación moderna. OAuth 2.0 "
            "proporciona un framework de autorización estándar mientras JWT (JSON Web Tokens) "
            "permite autenticación stateless. Cubrimos flujos de authorization code, "
            "client credentials, refresh tokens, mejores prácticas para almacenamiento "
            "seguro, prevención de ataques CSRF y XSS, y herramientas como Keycloak y Auth0."
        ),
        "url": "https://tech.example.com/api-security-oauth-jwt",
        "tags": ["seguridad", "oauth", "jwt", "api", "rest", "autenticacion"],
        "type": "documentation",
        "date": "2023-12-18",
        "popularity": 5500,
    },
    {
        "id": "doc_008",
        "title": "Arquitectura de Microservicios con Node.js",
        "content": (
            "Los microservicios permiten escalar y mantener aplicaciones complejas de forma "
            "independiente. Node.js con su modelo asíncrono basado en eventos es ideal para "
            "servicios ligeros y APIs de alta concurrencia. Diseñamos una arquitectura "
            "completa con service discovery usando Consul, comunicación asíncrona con Kafka, "
            "API gateway con Kong, y observabilidad con Prometheus y Grafana."
        ),
        "url": "https://tech.example.com/microservices-nodejs",
        "tags": ["nodejs", "microservicios", "kafka", "arquitectura", "backend"],
        "type": "article",
        "date": "2024-01-25",
        "popularity": 3900,
    },
    {
        "id": "doc_009",
        "title": "CI/CD con GitHub Actions: Automatización Completa",
        "content": (
            "GitHub Actions ha democratizado CI/CD para proyectos de cualquier tamaño. "
            "Aprende a crear workflows para testing automático, linting, build de Docker "
            "images, deployment a AWS/GCP/Azure, gestión de secretos, matrix builds para "
            "múltiples versiones de Python/Node, y cómo optimizar tiempos con caché de "
            "dependencias y ejecución paralela de jobs."
        ),
        "url": "https://tech.example.com/github-actions-cicd",
        "tags": ["github actions", "cicd", "devops", "automatizacion", "docker"],
        "type": "tutorial",
        "date": "2024-02-08",
        "popularity": 6700,
    },
    {
        "id": "doc_010",
        "title": "Sistemas de Recuperación de Información: LSI y Modelos Vectoriales",
        "content": (
            "La Indexación Semántica Latente (LSI) es un modelo no básico de recuperación "
            "de información que usa descomposición SVD para descubrir relaciones semánticas "
            "latentes entre términos y documentos. A diferencia del modelo vectorial clásico, "
            "LSI puede relacionar términos sinónimos y resolver polisemia. Implementaciones "
            "con scikit-learn TruncatedSVD y comparativa con modelos probabilísticos de "
            "lenguaje y redes neuronales para IR."
        ),
        "url": "https://tech.example.com/lsi-information-retrieval",
        "tags": ["lsi", "sri", "recuperacion informacion", "nlp", "svd", "vectores"],
        "type": "article",
        "date": "2024-03-05",
        "popularity": 2100,
    },
    {
        "id": "doc_011",
        "title": "PostgreSQL Avanzado: Índices, JSONB y Full-Text Search",
        "content": (
            "PostgreSQL es una de las bases de datos relacionales más potentes. Exploramos "
            "índices avanzados (GIN, GiST, BRIN), almacenamiento de datos semiestructurados "
            "con JSONB, búsqueda de texto completo con tsvector y tsquery, particionamiento "
            "de tablas, replicación streaming, y extensiones como PostGIS para datos "
            "geoespaciales y pg_vector para embeddings de IA."
        ),
        "url": "https://tech.example.com/postgresql-advanced",
        "tags": ["postgresql", "base de datos", "sql", "full text search", "ia"],
        "type": "documentation",
        "date": "2023-10-30",
        "popularity": 4800,
    },
    {
        "id": "doc_012",
        "title": "Rust para Desarrolladores de Python: Rendimiento sin Sacrificar Seguridad",
        "content": (
            "Rust ofrece rendimiento comparable a C++ con garantías de seguridad en memoria "
            "sin garbage collector. Para desarrolladores de Python, Rust es ideal cuando se "
            "necesita rendimiento crítico. Cubrimos ownership, borrowing, lifetimes, "
            "concurrencia sin data races, y cómo integrar Rust con Python mediante PyO3 "
            "para acelerar código crítico sin abandonar el ecosistema Python."
        ),
        "url": "https://tech.example.com/rust-for-python-developers",
        "tags": ["rust", "python", "rendimiento", "sistemas", "pyo3"],
        "type": "tutorial",
        "date": "2024-04-03",
        "popularity": 5100,
    },
]


# ---------------------------------------------------------------------------
# Queries de prueba con juicios de relevancia
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    {
        "query_id": "q1",
        "query": "machine learning python frameworks inteligencia artificial",
        "relevant": ["doc_001", "doc_004", "doc_010"],
        "grades": {"doc_001": 3, "doc_004": 2, "doc_010": 1},
    },
    {
        "query_id": "q2",
        "query": "docker kubernetes contenedores despliegue devops",
        "relevant": ["doc_002", "doc_005", "doc_009"],
        "grades": {"doc_002": 3, "doc_009": 2, "doc_005": 1},
    },
    {
        "query_id": "q3",
        "query": "bases de datos vectoriales embeddings rag chromadb",
        "relevant": ["doc_006", "doc_010", "doc_011"],
        "grades": {"doc_006": 3, "doc_010": 2, "doc_011": 1},
    },
    {
        "query_id": "q4",
        "query": "seguridad api rest autenticacion oauth",
        "relevant": ["doc_007", "doc_008"],
        "grades": {"doc_007": 3, "doc_008": 1},
    },
    {
        "query_id": "q5",
        "query": "javascript frontend frameworks react vue angular",
        "relevant": ["doc_003"],
        "grades": {"doc_003": 3},
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  SRI — Tecnología y Software")
    print("  Prueba de Integración Completa")
    print("=" * 60)

    # 1. Crear pipeline
    pipeline = SRIPipeline(lsi_components=50, top_k=5, load_existing=False)

    # 2. Indexar corpus
    pipeline.index(SAMPLE_DOCUMENTS, save=True)

    # 3. Guardar queries de prueba
    os.makedirs("data", exist_ok=True)
    test_path = "data/test_queries.json"
    Evaluator.save_test_queries(TEST_QUERIES, test_path)

    # 4. Búsquedas de ejemplo
    print("\n--- Búsqueda: 'machine learning python' ---")
    results = pipeline.search("machine learning python", top_k=5)
    for r in results:
        print(f"  [{r['position']}] ({r['final_score']:.4f}) "
              f"{r['title']} [{r['display_type']}]")

    print("\n--- Búsqueda: 'bases de datos vectoriales' ---")
    results = pipeline.search("bases de datos vectoriales embeddings", top_k=5)
    for r in results:
        print(f"  [{r['position']}] ({r['final_score']:.4f}) "
              f"{r['title']} [{r['display_type']}]")

    print("\n--- Búsqueda: 'seguridad api oauth jwt' ---")
    results = pipeline.search("seguridad api oauth jwt", top_k=5)
    for r in results:
        print(f"  [{r['position']}] ({r['final_score']:.4f}) "
              f"{r['title']} [{r['display_type']}]")

    # 5. Evaluación completa
    print("\n--- Evaluación del sistema ---")
    eval_output = "data/evaluation_report.json"
    pipeline.evaluate(test_path, output_path=eval_output)

    print("\n✓ Demo completado. Archivos generados en data/")


if __name__ == "__main__":
    main()
