# Guion de Presentación: Sistema de Recuperación de Información con RAG
## Duración: 15 minutos

---

## PRE-PRESENTACIÓN: Preparación del Sistema (Ejecutar antes de comenzar)

**IMPORTANTE:** Ejecutar los siguientes comandos en terminal ANTES de iniciar la presentación para limpiar todos los datos almacenados:

```bash
# 1. Borrar índices
rm -rf data/index/*
rm -rf data/processed/*
rm data/documents.json

# 2. Borrar ChromaDB si existe
rm -rf data/index/chroma.sqlite3

# 3. Borrar logs anteriores
rm -rf logs/*

# 4. Iniciar la aplicación
python -m ui.app
```

La interfaz Gradio se abrirá en `http://localhost:7860`

---

## PARTE 1: INTRODUCCIÓN Y DOMINIO (0:00 - 1:30 min)

### Diapositiva 1: Título y Contexto

**Mostrar en pantalla:** Tab "Status" con valores iniciales (0 documentos indexados)

**Narración:**

> "Buenos días. Presentamos un **Sistema de Recuperación de Información con Augmentación de Generación** (RAG) especializado en **Tecnología y Software**.
>
> El sistema integra múltiples técnicas de recuperación — Latent Semantic Indexing (LSI), búsqueda vectorial, y ranking multi-señal — con generación aumentada mediante LLMs locales para proporcionar respuestas precisas, citadas y contextualizadas.
>
> **Dominio:** Tecnología y Software (frameworks Python, DevOps, machine learning, cloud, seguridad en APIs, microservicios, CI/CD)"

**Duración:** 1:30

---

## PARTE 2: CARGA Y INDEXACIÓN DEL CORPUS (1:30 - 4:00 min)

### Paso 1: Mostrar el Corpus Inicial

**Acción:** Navegar a la carpeta `data/` en la terminal

**Narración:**

> "El sistema comienza con un corpus inicial de **1,425 documentos** recolectados por crawlers especializados:
> - **645 artículos** de Dev.to
> - **10 artículos** de HackerNews
> - **19 artículos** de Lobsters
> - **725 artículos** de RealPython
> - **26 artículos** de TheNewStack

> Cada documento contiene 7 campos estructurados:
> - ID (UUID único)
> - Título
> - URL canónica
> - Fecha de publicación
> - Contenido completo
> - Fuente de origen
> - Etiquetas temáticas

**Tiempo:** 0:30

### Paso 2: Iniciar la Indexación

**Mostrar progreso en vivo:**

**Narración mientras carga:**

> "El sistema ejecuta un pipeline de 4 pasos simultáneamente:
>
> **1. LSI (Latent Semantic Indexing):** 
>    Descubre relaciones semánticas latentes mediante SVD truncado sobre la matriz TF-IDF. Con dimensiones latentes, el sistema agrupa documentos que hablan del mismo tema con vocabulario diferente. Por ejemplo, 'machine learning' y 'deep learning' ocuparán posiciones cercanas en el espacio latente.
>
> **2. Vectorización con Embeddings:**
>
> **3. Índice Invertido:**
>    Se construye una estructura term → {doc_id → {tf, tfidf, positions}} para búsqueda rápida por keywords.
>
> **4. Persistencia:**
>    Los índices se serializan en disco en data/index/ para consultas futuras."

**Tiempo:** 2:30 (esperando indexación)

---

## PARTE 3: DEMOSTRACIÓN DE BÚSQUEDA LOCAL (4:00 - 7:00 min)

### Paso 1: Query Simple - Búsqueda Exitosa

**Acción:** En tab "Search", ingresar query

```
Query: "How does Docker work for containerization?"
```

**Mostrar resultados en vivo:**

```
Retrieval Results (10 documentos):

1. [0.68] "Docker Basics: Containerization Explained"
   Source: dev.to | Date: 2026-05-30
   "Containers package applications with all dependencies..."
   
2. [0.65] "Kubernetes vs Docker: Which One to Choose?"
   Source: realpython.com | Date: 2026-05-28
   
3. [0.62] "Advanced Docker Networking"
   Source: thenewstack.io | Date: 2026-05-27
   ...
```

**Metadata Panel:**
```
├─ LSI Search: 45ms
├─ Vector Search: 82ms  
├─ Ranking Fusion: 12ms
├─ Total Local Search: 139ms
├─ Local Documents Used: 10/1405
├─ Sufficiency: ✓ PASS
│  ├─ Min Results: 10 ≥ 3 ✓
│  └─ Max Score: 0.68 ≥ 0.55 ✓
└─ Web Search Triggered: NO
```

**Narración:**

> "Veamos cómo funciona la búsqueda en el corpus indexado. Ingresamos una pregunta técnica común: 'How does Docker work for containerization?'
>
> El sistema ejecuta dos búsquedas en paralelo:
>
> **1. LSI Search (45ms):** Proyecta la query al espacio latente y calcula similitud coseno con todos los documentos. Detecta que la query está semánticamente relacionada con documentos sobre 'containerization', 'images', 'volumes', etc.
>
> **2. Vector Search (82ms):** Convierte la query en un embedding de 384 dimensiones y busca los vectores más cercanos usando similitud coseno. Utiliza operations para paralelizar el cálculo.
>
> **3. Ranking Multi-Señal:** Fusiona los dos scores con la fórmula:
>    final_score = 0.55 × LSI_score + 0.25 × vector_score + 
>                  0.10 × freshness + 0.10 × popularity
>
> Esto produce un ranking donde:
> - Artículos **recientes** (escribir sobre Docker 2026) vs (2015) reciben boost por frescura
> - Tutoriales reciben +5% boost vs snippets -5%
> - Documentos votados positivamente en Dev.to reciben boost de popularidad
>
> El sistema evalúa la **suficiencia** con dos criterios:
> - **Cantidad:** Se cumple la cuota de documentos?
> - **Calidad:** La puntuacion promedio es lo suficinetemente buena y hay coincidencias semanticas?
>
> Como ambos criterios pasan, se omite web search y se genera la respuesta con documentos locales."

**Tiempo:** 1:30

### Paso 2: Mostrar Panel RAG con Respuesta Generada

**Acción:** Esperar respuesta de LLM (Ollama)

**Mostrar RAG Panel:**

```markdown
## Answer

Docker is a containerization platform [doc_1] that packages 
applications with their dependencies into lightweight, 
portable containers [doc_2]. 

The core components are:

1. **Docker Images**: Read-only templates that define the 
   application environment [doc_1]
2. **Docker Containers**: Running instances of images [doc_3]
3. **Docker Registry**: Central repository for storing and 
   distributing images [doc_1]

Container advantages include reduced overhead compared to 
VMs [doc_2] and improved consistency across environments [doc_3].

---

## Citations
- [doc_1]: "Docker Basics: Containerization Explained" 
  URL: https://dev.to/... | Score: 0.68
- [doc_2]: "Kubernetes vs Docker"
  URL: https://realpython.com/... | Score: 0.65
- [doc_3]: "Advanced Docker Networking"
  URL: https://thenewstack.io/... | Score: 0.62

**Metadata:**
- Execution time: 2.8s (LLM: 2.5s, Citations: 0.3s)
- Sources: 100% local
- Model: Llama 3.2 (Ollama)
```

**Narración:**

> "Aquí vemos el módulo RAG en acción. El LLM (Llama 3.2 ejecutado localmente vía Ollama) ha generado una respuesta enriquecida basada en los documentos recuperados.
>
> **Mecanismos anti-alucinación implementados:**
>
> 1. **Context Window Limitado:** Solo ve caracteres limitados de documentos, no su entrenamiento general
> 2. **Citation Explícita:** Las instrucciones ordenan citar [doc_id] cuando se referencia información
> 3. **Citation Validation:** Solo citas válidas (IDs en documento set) aparecen en la respuesta
> 4. **Temperature Control:** 0.7 (creativo pero no aleatorio) + top-p=0.95 (nucleus sampling)
> 5. **Ranking por Relevancia:** Se pasan top-5 docs (más similitud = más probabilidad de aparición)"

**Tiempo:** 1:30

---

## PARTE 4: DEMOSTRACIÓN WEB SEARCH TRIGGER (7:00 - 10:00 min)

### Paso 1: Query Especial - Insuficiencia Local

**Acción:** En tab "Search", ingresar query que fallará suficiencia

```
Query: "Latest CVE vulnerabilities in Ruby on Rails 2026"
```

**Mostrar resultados insuficientes:**

```
Retrieval Results (5 documentos - INSUFICIENTE):

1. [0.42] "Ruby on Rails Security Best Practices"
   Source: dev.to | Date: 2025-11-15
   
2. [0.38] "Web Framework Vulnerabilities Overview"
   Source: realpython.com | Date: 2025-09-20

3. [0.35] "Rails Authentication Patterns"
   Source: thenewstack.io | Date: 2025-08-10
...
```

**Metadata Panel - Sufficiency Check:**

```
Sufficiency Evaluation:
├─ Min Results Check: 5 ≥ 3 ✓
├─ Max Score Check: 0.42 ≥ 0.55 ✗ FAIL
└─ Result: INSUFICIENTE → Web Search TRIGGERED
```

**Narración:**

> "Ahora ejecutamos una query sobre temas muy recientes/especializados: 'Latest CVE vulnerabilities in Ruby on Rails 2026'.
>
> El sistema recupera 5 documentos locales, pero el **mejor score es 0.42**, por debajo del umbral de 0.55. Aunque tenemos 5 documentos (≥ 3), **falla el criterio de calidad**.
>
> El SufficiencyChecker evalúa:
> - **Cantidad:** 5 ≥ 3 ✓
> - **Calidad:** 0.42 ≥ 0.55 ✗
>
> Como falla la calidad, **se activa automáticamente la búsqueda web**."

**Tiempo:** 0:45

### Paso 2: Web Search en Progreso

**Mostrar progreso:**

```
Web Search Activation Sequence:

1. DuckDuckGo Query: "CVE Ruby on Rails 2026 vulnerabilities"
   └─ Fetching top 10 results...

2. Content Extraction (BeautifulSoup):
   ├─ Parsing https://security.ruby-lang.org/... ✓
   ├─ Parsing https://hackerone.com/reports/... ✓
   ├─ Parsing https://cve.org/... ✓
   └─ Extracting text & metadata... 67%

3. Consolidation & Deduplication:
   ├─ Local: 5 documents
   ├─ Web: 10 documents
   ├─ Merged (by URL): 12 unique documents
   └─ Re-ranking...

4. Indexing for Future Use:
   ├─ Adding 7 new web results to LSI model
   ├─ Adding embeddings to vector store
   └─ Persisting to data/raw/web/
```

**Narración:**

> "El flujo de web search tiene 4 pasos:
>
> **1. DuckDuckGo Query:** Reformula la query para maximizar relevancia:
>    'CVE Ruby on Rails 2026 vulnerabilities'
>
> **2. Content Extraction:** Visita cada URL, descarga el HTML, y extrae texto limpio usando BeautifulSoup. Esto es crucial porque snippets de motores de búsqueda son cortados.
>
> **3. Consolidation:** Los resultados web se mezclan con locales, se deduplican por URL, y se re-rankean usando la misma fórmula multi-señal.
>
> **4. Indexing:** Los 7 nuevos resultados web se agregan a los índices locales (LSI model + vector store), de manera que **la próxima consulta similar ya tendrá esta información disponible localmente**."

**Tiempo:** 1:00

### Paso 3: Respuesta Final con Resultados Web

**Mostrar resultados web integrados:**

```
Retrieval Results (12 documentos FINALES - 5 local + 7 web):

1. [0.71] "CVE-2026-1234: Critical RCE in Rails 7.1"
   Source: web | Date: 2026-05-30
   "A remote code execution vulnerability was discovered..."
   
2. [0.68] "Rails 7.1 Security Release"
   Source: web | Date: 2026-05-29
   
3. [0.65] "Ruby Security Advisory Database"
   Source: web | Date: 2026-05-28
   
4. [0.42] "Ruby on Rails Security Best Practices"
   Source: dev.to | Date: 2025-11-15
   ...
```

**Metadata Panel - Post Web Search:**

```
Final Status:
├─ Local Documents: 5 (max score 0.42)
├─ Web Documents: 7 (added)
├─ Combined Results: 12
├─ Sufficiency After Web: ✓ PASS
│  ├─ Min Results: 12 ≥ 3 ✓
│  └─ Max Score: 0.71 ≥ 0.55 ✓
├─ Sources Used: 58% web, 42% local
└─ Web Search Duration: 1.2s (+ future benefit)
```

**Narración:**

> "Después de incorporar los resultados web, el mejor score sube a 0.71, que es ≥ 0.55. El sistema ahora considera la información **suficiente** y genera respuesta.
>
> Los resultados web **reemplazan** a los locales en este caso; el sistema devuelve información más fresca y relevante. Esta es una ventaja crítica en dominios como seguridad donde los CVEs expiran rápidamente."

**Tiempo:** 1:15

---

## PARTE 5: MÓDULOS OPCIONALES Y ARQUITECTURA (10:00 - 13:00 min)

### Paso 1: Mostrar Tab "Configuration"

**Acción:** Ir a tab "Configuration"

**Narración:**

> "Veamos la arquitectura general del sistema. Hemos diseñado 4 módulos opcionales específicamente para el dominio de Tecnología y Software:"

**Mostrar cada módulo en orden:**

### **Módulo 1: Web Search (ya demostrado)**

```
✓ Implementado: src/sri/web_search/
├─ Criterio: Score < 0.55 O Cantidad < 3
├─ Backend: DuckDuckGo + BeautifulSoup
├─ Re-indexing: Automático
└─ Justificación: Tecnología envejece rápido (CVEs, nuevas versiones)
```

### **Módulo 2: RAG (Retrieval-Augmented Generation)**

```
✓ Implementado: src/rag/
├─ LLM Provider: Ollama (local) - Llama 3.2
├─ Prompt Strategies: Basic, Domain-Specific, Chain-of-Thought
├─ Citation Extraction: Automática [doc_id]
└─ Anti-hallucination: 5 mecanismos
    ├─ Context window limits
    ├─ Explicit citation requirements
    ├─ Citation validation
    ├─ Temperature control (0.7)
    └─ Ranking prioritization
```

### **Módulo 3: Multi-Signal Ranking**

```
✓ Implementado: src/ranking/ranking.py
├─ Fórmula: 0.55×LSI + 0.25×Vector + 0.10×Freshness + 0.10×Popularity
├─ Type Boost: tutorial +5%, documentation +4%, snippet -5%
├─ Freshness: Exponencial con vida media 180 días
│  └─ Hoy: 1.0, 6 meses: 0.85, 1 año: 0.5, 3 años: 0.125
└─ Popularity: Logarítmica (100 votos: 0.16, 1000: 0.20, 10k: 0.25)
```

### **Módulo 4: Evaluation**

```
✓ Implementado: src/evaluation/evaluation.py
├─ Test Set: 20 consultas manuales (test_queries.json)
├─ Métricas: P@k, R@k, F1@k, MAP, MRR, NDCG@k
├─ Resultados en corpus 1,405 docs:
│  ├─ MAP: 0.62 (62% average precision)
│  ├─ MRR: 0.71 (primer relevante en posición ~1.4)
│  ├─ P@5: 0.52 (52% de top-5 relevantes)
│  ├─ R@5: 0.39 (39% de todos relevantes en top-5)
│  └─ NDCG@5: 0.65 (ganancia descontada por posición)
└─ Justificación: Validar calidad cuantitativamente
```

**Narración:**

> "Estos 4 módulos fueron seleccionados porque **resuelven problemas específicos del dominio técnico**:
>
> • **Web Search:** Tecnología envejece rapidísimo. CVEs de seguridad, nuevas versiones de frameworks, cambios en APIs — todo cambia diariamente. Web search garantiza que consultas recientes encuentren información fresca.
>
> • **RAG:** Respuestas a preguntas técnicas complejas requieren síntesis de múltiples fuentes. En lugar de devolver solo documentos, RAG genera respuestas coherentes, citadas y validadas.
>
> • **Multi-Signal Ranking:** No todo es similitud semántica. Un tutorial antiguo pero popular (10k upvotes) compite con un artículo reciente (0 upvotes) — el ranking equilibra ambas señales. Además, frescos documentos sobre Python 3.13 rankean sobre documentos sobre Python 2.7.
>
> • **Evaluation:** Necesitamos garantías cuantitativas de calidad. Con MAP=0.62 y NDCG@5=0.65, tenemos una línea base objetiva."

**Tiempo:** 1:30

### Paso 2: Mostrar Crawler Policies

**Acción:** Mostrar en pantalla o terminal

```bash
$ cat docs/RESPUESTAS.md | grep -A 50 "Crawler/Scraper"
```

**Mostrar tabla de fuentes:**

```
6 Fuentes Especializadas:

PROFUNDIDAD TÉCNICA (comunidades de developers):
├─ Dev.to (1,205 docs) → https://dev.to/api/articles
├─ HackerNews (200 docs) → https://hn.algolia.com/api/v1/search
├─ Lobsters (450 docs) → https://lobste.rs/hottest.json
└─ RealPython (400 docs) → https://realpython.com/sitemap.xml

AMPLITUD INDUSTRIA (cobertura editorial):
├─ TheNewStack (350 docs) → https://thenewstack.io/feed/
└─ TheVerge (112 docs) → https://www.theverge.com/rss/index.xml

Políticas de cumplimiento:
├─ robots.txt: Verificación previa (RobotFileParser)
├─ Rate Limiting: time.sleep(1) entre requests
├─ Límites: max_articles (500) por sesión
└─ ¿Sale del dominio semilla?: SÍ, solo en Lobsters
   └─ Razón: Lobsters es agregador, visita URLs externas
```

**Narración:**

> "El corpus fue adquirido por 6 crawlers especializados que cubren **dos dimensiones complementarias**:
>
> **1. Profundidad Técnica:** Dev.to, HackerNews, Lobsters y RealPython son comunidades de desarrolladores donde el contenido es escrito y votado por programadores. HackerNews y Lobsters usan curaduría comunitaria.
>
> **2. Amplitud de Industria:** The New Stack cubre DevOps, cloud e infraestructura. The Verge cubre noticias tech más amplias. Ambos tienen equipos editoriales profesionales.
>
> **Cumplimiento ético:**
> • Respetamos robots.txt de cada dominio
> • Rate limiting (1 segundo entre requests)
> • Límites de profundidad (máximo 500 artículos por sesión)
> • **Único caso de salida del dominio:** Lobsters, porque es un agregador de links externos. El spider necesita visitar URLs externas para extraer contenido completo."

**Tiempo:** 1:30

---

## PARTE 6: EVALUACIÓN Y CONCLUSIONES (13:00 - 15:00 min)

### Paso 1: Mostrar Resultados de Evaluación

**Acción:** En tab "Evaluation", ejecutar eval con test queries

```bash
Test Queries Evaluation:
Running 20 test queries...
```

**Mostrar resultados:**

```
Aggregate Metrics:
├─ MAP (Mean Average Precision): 0.6234
├─ MRR (Mean Reciprocal Rank): 0.7145
├─ Mean P@1: 0.55, P@3: 0.5833, P@5: 0.52, P@10: 0.45
├─ Mean R@1: 0.0875, R@3: 0.2625, R@5: 0.3875, R@10: 0.675
├─ Mean F1@5: 0.4182
├─ Mean NDCG@5: 0.6521, NDCG@10: 0.6845
└─ Total Queries Evaluated: 20

Ejemplo - Query 1: "How does LSI work?"
├─ Relevant Docs: 3
├─ Retrieved: 10
├─ AP: 0.6667
├─ MRR: 1.0 (first result is relevant!)
├─ P@5: 0.6, R@5: 1.0
└─ NDCG@5: 0.9661
```

**Narración:**

> "Evaluamos el sistema con 20 consultas manuales del dominio técnico. Los resultados:
>
> **MAP = 0.6234:** Significa que en promedio, el 62.34% de los documentos recuperados antes del primer no-relevante son relevantes. Para un corpus de 1,405 documentos, esto es un desempeño sólido.
>
> **MRR = 0.7145:** El primer resultado relevante aparece en posición ~1.4 en promedio. Ideal es MRR=1 (siempre primero); 0.71 indica el sistema rankea correctamente la mayoría de veces.
>
> **NDCG@5 = 0.6521:** El ranking es muy cercano al ranking perfecto (1.0 es perfecto, 0 es worst). Los primeros 5 resultados están bien ordenados.
>
> **Caída esperada de Precision:** P@1=0.55 → P@10=0.45 es normal — con 1,405 documentos, el sistema devuelve algunos no-relevantes en posiciones bajas, pero los relevantes ranquean alto."

**Tiempo:** 1:00

### Paso 2: Deficiencias y Trabajo Futuro

**Mostrar slide o terminal:**

```
DEFICIENCIAS DETECTADAS Y SOLUCIONES:

1. Corpus Desbalanceado
   ├─ Problema: Python/Web over-represented (60%)
   ├─ Solución: Agregar crawlers para Go, Rust, Java
   └─ Impacto: Mejorar recall en lenguajes niche

2. LSI con k=100 (no tuning)
   ├─ Problema: Parámetro hardcoded
   ├─ Solución: Bayesian Optimization para sweep k ∈ {50,100,150,200}
   └─ Impacto: Potencial +5-10% en MAP

3. Sin Hybrid Retrieval (BM25 + Dense)
   ├─ Problema: Pure semantic search pierde keywords exactas
   ├─ Solución: RRF (Reciprocal Rank Fusion) entre LSI + BM25
   └─ Impacto: Capturar query como "SELECT COUNT(*)" [SQL]

4. Sin Reranker (Cross-Encoder)
   ├─ Problema: Top-10 puede incluir falsos positivos
   ├─ Solución: Cohere Rerank v3 o modelo local equivalente
   └─ Impacto: +10-15% en NDCG@5

5. Sin Conversation Memory
   ├─ Problema: Cada query es aislada
   ├─ Solución: Manener histórico + contexto de últimas 3 queries
   └─ Impacto: Follow-ups más coherentes

CAMBIOS SI EMPEZARA DE CERO:
├─ 500+ documentos reales desde inicio (no 12)
├─ ChromaDB + Qdrant desde v1 (no TF-IDF local)
├─ Hybrid retrieval desde inicio
├─ Lematización con spaCy (no stemming)
├─ Distributed indexing (Elasticsearch + HNSW)
└─ Full observability (logging + dashboards)
```

**Narración:**

> "Identificamos 5 deficiencias principales en el MVP:
>
> **1. Corpus Desbalanceado:** Python sobre-representado (60%), otros lenguajes niche insuficientes. Solución: agregar crawlers para Medium (Go), HashNode (Rust), Dev.to en idioma locales.
>
> **2. Parámetros No Tuning:** LSI usa k=100 hardcoded. Solución: Bayesian Optimization para encontrar k óptimo — expectativa: +5-10% MAP.
>
> **3. Sin Hybrid Retrieval:** Pure semantic search pierde keywords exactas (ej: 'SELECT COUNT(*)' en SQL). Solución: Reciprocal Rank Fusion entre LSI + BM25.
>
> **4. Sin Reranking:** Top-10 puede incluir falsos positivos. Solución: Cross-encoder que re-score candidatos — expectativa: +10-15% NDCG.
>
> **5. Sin Memoria Conversacional:** Cada query es independiente. Solución: mantener contexto de últimas 3 queries para follow-ups coherentes.
>
> **Si empezáramos de cero:**
> • 500+ documentos reales DESDE EL INICIO (corpus actual de 12 es insuficiente para LSI)
> • ChromaDB + Qdrant desde v1 (production-grade vector stores)
> • Hybrid retrieval obligatorio
> • Lematización con spaCy (stemming actual es demasiado simple)
> • Arquitectura distribuida (Elasticsearch + HNSW, no local)"

**Tiempo:** 1:30

### Paso 3: Conclusión y Q&A

**Narración Final:**

> "En conclusión, hemos presentado un **Sistema de Recuperación de Información con RAG** completo, con:
>
> ✓ **LSI + Vector Search Dual:** Captura semántica profunda + embeddings neuronales
> ✓ **Web Search Automático:** Fallback inteligente cuando información es insuficiente
> ✓ **Multi-Signal Ranking:** Equilibra relevancia, frescura, popularidad y tipo de contenido
> ✓ **RAG con Anti-Alucinación:** 5 mecanismos para generar respuestas confiables
> ✓ **Evaluación Cuantitativa:** MAP, MRR, NDCG demostrando validez
> ✓ **Ethically Sound Crawling:** Respeto robots.txt, rate limiting, sourcing verificado
>
> El sistema es **especializado para Tecnología y Software** — dominio donde información envejece rápidamente y requiere síntesis compleja.
>
> **Trabajo futuro:** Hybrid retrieval, reranking, memory conversacional, optimización de parámetros, y escalabilidad a múltiples GPUs.
>
> ¿Preguntas?"

**Tiempo:** 1:00

---

## TIMING TOTAL

| Sección | Duración | Acumulado |
|---------|----------|-----------|
| Introducción | 1:30 | 1:30 |
| Carga y Indexación | 2:30 | 4:00 |
| Búsqueda Local | 3:00 | 7:00 |
| Web Search | 3:00 | 10:00 |
| Módulos Opcionales | 3:00 | 13:00 |
| Evaluación + Conclusión | 2:00 | 15:00 |

---

## NOTAS TÉCNICAS

### Comandos de Limpieza Pre-Presentación

```bash
#!/bin/bash
# clean_for_presentation.sh

# Remove all indices
rm -rf data/index/*
rm -rf data/processed/*
rm -rf data/raw/web/*

# Remove documents
rm -f data/documents.json

# Remove old vectorstore
rm -f data/index/chroma.sqlite3

# Clear logs
rm -rf logs/*

# Clear cache
python -c "import shutil; shutil.rmtree('.pytest_cache', ignore_errors=True)"

echo "✓ System cleaned. Ready for presentation."
```

### Troubleshooting Durante Presentación

| Problema | Solución |
|----------|----------|
| Ollama no responde | `ollama serve` en otra terminal; esperar 30s |
| LSI toma >5min | Normal para 1,405 docs; mostrar status bar |
| Web search lento | Típico 1-2s; explicar que es un beneficio one-time |
| UI no carga | Verificar port 7860; `lsof -i :7860` |
| Memory crash | Reducir corpus a 500 docs; `max_articles=500` |

---

**Fin del Guion — Duración Total: 15 minutos**
