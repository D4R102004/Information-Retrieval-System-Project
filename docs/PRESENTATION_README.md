# Guía de Presentación - Sistema de Recuperación de Información con RAG

## 📋 Documentos Principales

### 1. **RESPUESTAS.md** (Lectura Previa)
Análisis técnico completo de 16 preguntas fundamentales sobre el proyecto:
- Dominio temático y corpus
- Modelo LSI y ventajas
- Módulos opcionales (Web Search, RAG, Ranking, Evaluation)
- Crawlers y políticas de cumplimiento
- Arquitectura end-to-end
- Resultados cuantitativos
- Deficiencias y soluciones propuestas

**Ubicación:** `docs/RESPUESTAS.md`
**Lectura recomendada:** Antes de la presentación (30 minutos)

### 2. **GUION.md** (Script de Presentación)
Script detallado de 15 minutos con:
- Timings exactos por sección
- Comandos a ejecutar en la UI
- Narración palabra por palabra
- Ejemplos de output esperado
- Troubleshooting

**Ubicación:** `docs/GUION.md`
**Referencia durante:** Presentación en vivo

---

## 🧹 Preparación Pre-Presentación

### Paso 1: Limpiar Sistema

**En macOS/Linux:**
```bash
bash clean_for_presentation.sh
```

**En Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File clean_for_presentation.ps1
```

**Resultado esperado:**
```
✓ Índices eliminados
✓ ChromaDB eliminado
✓ Resultados web eliminados
✓ Logs limpios
✓ Caché de Python limpio
✓ Directorios recreados

Sistema Limpio - LISTO PARA PRESENTACIÓN
```

### Paso 2: Iniciar Aplicación

```bash
python -m ui.app
```

**Verificar en navegador:** http://localhost:7860

---

## 🚀 Flujo de Presentación (15 minutos)

### PARTE 1: Introducción (0:00 - 1:30)
- Mostrar tab "Status" (0 documentos indexados)
- Explicar dominio: Tecnología y Software
- Mencionar corpus inicial: 1,405 documentos

### PARTE 2: Carga y Indexación (1:30 - 4:00)
- Tab "Configuration"
- Clic "Load Documents from Crawlers"
- Seleccionar "Force re-index corpus"
- **Mostrar progreso:**
  - LSI: TF-IDF → SVD 100-dims (k=100)
  - Vector: all-MiniLM-L6-v2 (384-dims)
  - Indexación persistente
- **Duración esperada:** ~2:30

### PARTE 3: Búsqueda Local Exitosa (4:00 - 7:00)
- Tab "Search"
- Query: `"How does Docker work for containerization?"`
- Mostrar 10 documentos recuperados con scores
- Explicar:
  - LSI Search (45ms)
  - Vector Search (80ms)
  - Multi-Signal Ranking
  - Sufficiency Check: PASS
- Esperar respuesta RAG
- Mostrar citas extraídas automáticamente

### PARTE 4: Web Search Trigger (7:00 - 10:00)
- Query: `"Latest CVE vulnerabilities in Ruby on Rails 2026"`
- Mostrar **insuficiencia local** (score < 0.55)
- Web Search se activa automáticamente
- Mostrar consolidación y re-indexación
- Respuesta final con resultados web

### PARTE 5: Módulos Opcionales (10:00 - 13:00)
- Tab "Configuration"
- Explicar 4 módulos opcionales:
  1. **Web Search:** Fallback automático
  2. **RAG:** Generación + citations
  3. **Multi-Signal Ranking:** Fórmula 0.55×LSI + 0.25×Vector + ...
  4. **Evaluation:** MAP, MRR, NDCG
- Mostrar tabla de crawlers y políticas

### PARTE 6: Evaluación y Q&A (13:00 - 15:00)
- Tab "Evaluation"
- Ejecutar eval: `python test_eval.py`
- Mostrar resultados:
  - MAP: 0.6234
  - MRR: 0.7145
  - NDCG@5: 0.6521
- Mencionar deficiencias y soluciones futuras
- Abrir para preguntas

---

## 📊 Queries de Demostración Recomendadas

### Query 1: Búsqueda Simple (Exitosa)
```
"How does Docker work for containerization?"
```
**Resultado esperado:** 10 documentos, score máximo 0.68+, SUFFICIENCY PASS

### Query 2: Insuficiencia de Información
```
"Latest CVE vulnerabilities in Ruby on Rails 2026"
```
**Resultado esperado:** Score < 0.55, WEB SEARCH TRIGGERED

### Query 3: Técnica Específica
```
"Explain LSI and latent semantic indexing"
```
**Resultado esperado:** Auto-referencia (sistema recupera documentación propia)

### Query 4: Integración Compleja
```
"Best practices for Python microservices architecture with Kubernetes"
```
**Resultado esperado:** Multi-document synthesis en RAG

---

## ⚙️ Configuración de Parámetros UI

### Tab "Search Options"
- **Use web search:** ✓ (enabled)
- **Auto-reload database:** ✓ (enabled)
- **Max local results:** 5
- **Max web results:** 10

### Tab "Configuration"
- **Min documents threshold:** 3
- **Score threshold:** 0.55
- **LSI components:** 100
- **Embedding model:** all-MiniLM-L6-v2

---

## 🔍 Posibles Problemas y Soluciones

### Problema: Ollama no responde
**Síntoma:** "Connection refused" al generar RAG
**Solución:** 
```bash
# En otra terminal
ollama serve

# Esperar 30 segundos hasta que esté listo
```

### Problema: Indexación tarda > 5 minutos
**Síntoma:** LSI training muy lento
**Solución:** Normal para 1,405 documentos. Mostrar la barra de progreso mientras se explica LSI.

### Problema: Web search muy lento
**Síntoma:** Respuesta web tarda > 2 segundos
**Solución:** Normal (fetching + parsing URLs). Explicar que es beneficio "one-time" — siguiente query similar será local.

### Problema: UI no carga
**Síntoma:** Navegador no alcanza localhost:7860
**Solución:** 
```bash
# Verificar puerto
lsof -i :7860  # macOS/Linux
netstat -ano | findstr :7860  # Windows

# Cambiar puerto en código si es necesario
```

### Problema: Memory crash
**Síntoma:** Python process muere durante indexación
**Solución:** Reducir corpus a 500 docs. Editar `src/main_orchestator.py`:
```python
MAX_ARTICLES = 500  # en lugar de 1405
```

---

## 📱 Navegación UI

```
┌─────────────────────────────────────────────┐
│ [Search] [Configuration] [Evaluation] [Status]
├─────────────────────────────────────────────┤
│                                             │
│ SEARCH TAB (principal):                     │
│ ├─ Query input box                          │
│ ├─ Search / Clear buttons                   │
│ ├─ Advanced options (accordion)             │
│ └─ Results panels (3 columnas):             │
│    ├─ Left: Retrieved documents             │
│    ├─ Center: RAG answer                    │
│    └─ Right: Metadata + citations           │
│                                             │
│ CONFIGURATION TAB:                          │
│ ├─ Load Documents button                    │
│ ├─ Force re-index checkbox                  │
│ ├─ Thresholds (sliders)                     │
│ └─ Model selection dropdowns                │
│                                             │
│ EVALUATION TAB:                             │
│ ├─ Run Evaluation button                    │
│ ├─ Results display (metrics)                │
│ └─ Export results                           │
│                                             │
│ STATUS TAB:                                 │
│ ├─ Documents indexed: N                     │
│ ├─ Last update: timestamp                   │
│ └─ System health                            │
└─────────────────────────────────────────────┘
```

---

## 📚 Documentación Referencia

- **docs/RESPUESTAS.md** — Análisis técnico completo (16 preguntas)
- **docs/GUION.md** — Script de presentación (15 minutos)
- **docs/INDEX.md** — Estructura general del proyecto
- **docs/ARCHITECTURE_ANALYSIS.md** — Análisis profundo de arquitectura
- **docs/PRE_RAG_STATUS.md** — Estado pre-RAG
- **docs/RAG_IMPLEMENTATION_PLAN.md** — Plan de implementación

---

## 🎯 Objetivos de Presentación

✓ Demostrar funcionamiento completo del sistema  
✓ Responder todas las 16 preguntas  
✓ Mostrar indexación del corpus inicial  
✓ Visualizar búsqueda local (LSI + Vector)  
✓ Activar web search automático  
✓ Generar respuesta RAG con citas  
✓ Presentar módulos opcionales  
✓ Mostrar resultados de evaluación  
✓ Discutir deficiencias y futuro  
✓ Mantener tiempo total < 15 minutos  

---

## 💾 Cleanup Post-Presentación

```bash
# Opcional: guardar datos de presentación
cp -r data/evaluation demo_data/

# Limpiar para siguiente demo
bash clean_for_presentation.sh
```

---

**Última actualización:** Junio 1, 2026  
**Versión:** 1.0  
**Duración presentación:** 15 minutos  
**Sistema:** Listo para demostración
