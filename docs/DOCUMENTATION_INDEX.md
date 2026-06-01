# Índice de Documentación — Sistema de Recuperación de Información con RAG

## 📂 Estructura de Documentos

### 1. **RESPUESTAS.md** — Análisis Técnico Completo
**Propósito:** Responder las 16 preguntas fundamentales del proyecto  
**Longitud:** ~2,500 líneas  
**Audiencia:** Evaluadores académicos, técnicos, tribunal  
**Contenido:**

```
├─ Resumen Ejecutivo
├─ Pregunta 1: Dominio temático y características de documentos
├─ Pregunta 2: Modelo LSI (Latent Semantic Indexing)
├─ Pregunta 3: Módulos opcionales (Web Search, RAG, Ranking, Eval)
├─ Pregunta 4: Crawlers (fuentes, políticas, URLs semilla)
├─ Pregunta 5: Corpus indexado (cantidad, tipos, representatividad)
├─ Pregunta 6: Indexación (estructura, normalización, TF-IDF)
├─ Pregunta 7: Embeddings (generación, almacenamiento, búsqueda)
├─ Pregunta 8: Detección de insuficiencia (criterios 2-level)
├─ Pregunta 9: Flujo end-to-end (8 módulos, timings)
├─ Pregunta 10: Módulo RAG (demostración, anti-alucinación)
├─ Pregunta 11: Web search automático (trigger, consolidación)
├─ Pregunta 12: Interfaz visual (layout, decisiones de diseño)
├─ Pregunta 13: Ranking multi-señal (LSI + Vector + Frescura + Popularidad)
├─ Pregunta 14: Expansión/Feedback (no implementados, plantillas)
├─ Pregunta 15: Evaluación (MAP, MRR, NDCG, resultados)
├─ Pregunta 16: Deficiencias y resoluciones
└─ Referencias bibliográficas
```

**Cómo usar:** 
- Pre-lectura recomendada antes de presentación
- Referencia durante Q&A
- Base para documentación académica

---

### 2. **GUION.md** — Script de Presentación (15 min)
**Propósito:** Guiar la presentación en vivo  
**Duración:** 15 minutos exactos  
**Audiencia:** Presentador + audiencia en vivo  
**Estructura:**

```
├─ PRE-PRESENTACIÓN (Preparación)
│  └─ Comandos limpieza + verificación
├─ PARTE 1: Introducción (1:30)
│  └─ Dominio, corpus inicial, contexto
├─ PARTE 2: Carga y Indexación (2:30)
│  ├─ LSI training (visualización progreso)
│  └─ Vector embeddings generation
├─ PARTE 3: Búsqueda Local (3:00)
│  ├─ Query exitosa
│  └─ Explicación flujo LSI + Vector + Ranking
├─ PARTE 4: Web Search Trigger (3:00)
│  ├─ Insuficiencia local
│  └─ Consolidación web results
├─ PARTE 5: Módulos Opcionales (3:00)
│  ├─ Web Search
│  ├─ RAG
│  ├─ Ranking
│  └─ Evaluation
├─ PARTE 6: Resultados + Q&A (2:00)
│  └─ MAP, MRR, NDCG + deficiencias
└─ TIMING TOTAL: 15:00
```

**Cómo usar:**
- Leer durante presentación (ayuda memoria)
- Seguir timings exactos
- Usar queries de ejemplo proporcionadas
- Referencia narración palabra por palabra

---

### 3. **PRESENTATION_README.md** — Guía Operativa
**Propósito:** Instrucciones paso a paso para ejecutar presentación  
**Longitud:** ~500 líneas  
**Audiencia:** Presentador (técnico)  
**Contenido:**

```
├─ Documentos principales
├─ Preparación pre-presentación
│  ├─ Limpiar sistema (bash/PowerShell)
│  └─ Iniciar aplicación
├─ Flujo de presentación (15 min)
│  └─ 6 partes con timings
├─ Queries de demostración recomendadas
├─ Configuración parámetros UI
├─ Troubleshooting posibles problemas
├─ Navegación UI (diagrama)
├─ Documentación referencia
├─ Objetivos de presentación (checklist)
└─ Cleanup post-presentación
```

**Cómo usar:**
- Ejecutar antes de presentación (30 min antes)
- Referencia durante si algo no funciona
- Problemas + soluciones incluidos

---

### 4. **clean_for_presentation.sh** — Script Limpieza (macOS/Linux)
**Propósito:** Borrar todos los datos antes de demostración  
**Tiempo ejecución:** < 30 segundos  
**Acciones:**

```bash
rm -rf data/index/*           # Índices LSI + Vector
rm -rf data/processed/*       # Documentos procesados
rm -f data/documents.json     # Corpus consolidado
rm -f data/index/chroma.sqlite3  # ChromaDB
rm -rf logs/*                 # Logs anteriores
# + limpieza caché Python
# + recreación directorios
```

**Cómo usar:**
```bash
bash clean_for_presentation.sh
```

---

### 5. **clean_for_presentation.ps1** — Script Limpieza (Windows)
**Propósito:** Equivalente PowerShell del script de limpieza  
**Tiempo ejecución:** < 30 segundos  
**Cómo usar:**
```powershell
powershell -ExecutionPolicy Bypass -File clean_for_presentation.ps1
```

---

## 🎯 Flujo de Uso Recomendado

### Para Preparación (día anterior)
1. Leer **RESPUESTAS.md** completo (entender proyecto)
2. Leer **GUION.md** (familiarizarse con narración)
3. Revisar **PRESENTATION_README.md** (checklist)

### Para Ejecución (30 min antes)
1. Ejecutar script limpieza (`clean_for_presentation.sh` o `.ps1`)
2. Verificar que `python -m ui.app` funciona
3. Abrir http://localhost:7860
4. Revisar queries de demostración

### Durante Presentación (15 min)
1. Seguir **GUION.md** paso a paso
2. Usar **PRESENTATION_README.md** si hay problemas
3. Mostrar **RESPUESTAS.md** si hay preguntas técnicas

### Post-Presentación
1. Guardar datos si es necesario: `cp -r data/evaluation demo_data/`
2. Ejecutar limpieza de nuevo para siguiente demo

---

## 📊 Relación entre Documentos

```
RESPUESTAS.md (16 preguntas)
    ↓
GUION.md (Responde 16 preguntas EN VIVO)
    ↓
PRESENTATION_README.md (Cómo ejecutar GUION.md)
    ↓
clean_for_presentation.* (Preparación del GUION.md)
```

**RESPUESTAS.md** → Teoría profunda  
**GUION.md** → Demostración práctica  
**PRESENTATION_README.md** → Procedimientos operativos  
**Scripts** → Automatización  

---

## 🎓 Para Diferentes Audiencias

### Tribunal Académico
> "Léan **RESPUESTAS.md** para evaluación. Es la fuente de verdad técnica."

### Presentador
> "Sigan **GUION.md** durante la presentación. Timings exactos + narración incluidos."

### Técnicos Verificadores
> "Usen **PRESENTATION_README.md** para reproducir setup y observar ejecución."

### Autores (Team)
> "Revisar todos los documentos para validar consistencia antes de presentar."

---

## ✅ Checklist Pre-Presentación

- [ ] Leer RESPUESTAS.md (entender técnica)
- [ ] Leer GUION.md (memorizar flujo)
- [ ] Revisar PRESENTATION_README.md (procedimientos)
- [ ] Ejecutar clean_for_presentation.sh/ps1
- [ ] Verificar Ollama está disponible (`ollama serve`)
- [ ] Verificar UI carga (http://localhost:7860)
- [ ] Probar query 1 localmente: "Docker containerization"
- [ ] Probar query 2 insuficiencia: "Latest CVE Ruby 2026"
- [ ] Verificar timings (15 min máximo)
- [ ] Preparar respuestas a preguntas técnicas (de RESPUESTAS.md)

---

## 📞 Contacto y Soporte

**Documentación:** Todos los archivos están en `docs/`  
**Scripts:** Scripts de limpieza en raíz del proyecto  
**Preguntas técnicas:** Ver RESPUESTAS.md (secciones 1-16)  
**Problemas ejecución:** Ver PRESENTATION_README.md (troubleshooting)  

---

**Versión:** 1.0  
**Fecha:** Junio 1, 2026  
**Estado:** Listo para presentación  
**Duración presentación:** 15 minutos (exacto)
