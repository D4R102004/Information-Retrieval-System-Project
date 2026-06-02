### Leyenda del módulo de Evaluación

El módulo de **Evaluación** mide qué tan bien el sistema recupera documentos relevantes para un conjunto de consultas de prueba. Cada consulta incluye una pregunta y una lista de documentos considerados correctos o relevantes. El evaluador ejecuta la búsqueda, compara los documentos recuperados contra esa lista esperada y calcula métricas estándar de Recuperación de Información.

#### Conceptos básicos

- **Consulta de prueba**: pregunta usada para evaluar el sistema.
- **Documento relevante**: documento que debería aparecer como respuesta correcta para esa consulta.
- **Documento recuperado**: documento que el sistema devuelve después de ejecutar la búsqueda.
- **Ranking**: orden en que aparecen los documentos recuperados. Los primeros puestos son los más importantes.
- **k**: cantidad de primeros resultados que se analizan. Por ejemplo, `@5` significa “en los primeros 5 resultados”.

#### Métricas principales

| Métrica | Qué significa | Cómo interpretarla |
|---|---|---|
| **Precision@k / P@k** | De los primeros `k` documentos recuperados, qué proporción era relevante. | Alta precisión significa que los primeros resultados tienen poco “ruido”. |
| **Recall@k / R@k** | De todos los documentos relevantes esperados, qué proporción fue encontrada en los primeros `k` resultados. | Alto recall significa que el sistema está encontrando la mayor parte de lo importante. |
| **F1@k** | Promedio balanceado entre Precision@k y Recall@k. | Es útil cuando quieres una medida única que combine calidad y cobertura. |
| **MRR** | Posición del primer documento relevante encontrado. | Si el primer resultado ya es relevante, el MRR es alto. Si aparece muy abajo, el MRR baja. |
| **AP** | Calidad del ranking para una consulta, considerando en qué posiciones aparecen los documentos relevantes. | Premia que los documentos relevantes aparezcan pronto y no al final. |
| **MAP** | Promedio del AP en todas las consultas evaluadas. | Resume el rendimiento global del sistema en todo el conjunto de pruebas. |
| **NDCG@k** | Calidad del orden de los primeros `k` resultados usando grados de relevancia. | Es especialmente útil cuando algunos documentos son más relevantes que otros. |

#### Escala de valores

Todas estas métricas normalmente se interpretan entre **0 y 1**:

- **0.00**: el sistema no recuperó documentos relevantes dentro del rango evaluado.
- **0.50**: rendimiento intermedio.
- **1.00**: resultado ideal para esa métrica.

#### Por qué pueden salir métricas en cero

Si todas las métricas salen en `0`, no necesariamente significa que la UI esté rota. Normalmente significa que los IDs de documentos marcados como relevantes **no aparecen dentro de los primeros resultados recuperados**. Las causas más comunes son:

1. Las consultas de prueba son demasiado generales.
2. Los IDs relevantes no coinciden exactamente con los IDs reales de los documentos.
3. El documento correcto aparece, pero más abajo del límite evaluado, por ejemplo después del top 10.
4. El índice local, LSI o vector store no está cargado o no contiene esos documentos.

#### Cómo usar esta información

- Usa **Precision@k** para saber si los primeros resultados son limpios y relevantes.
- Usa **Recall@k** para saber si el sistema está encontrando todos los documentos importantes.
- Usa **MRR** para saber qué tan rápido aparece el primer resultado correcto.
- Usa **MAP** para comparar el rendimiento general entre versiones del sistema.
- Usa **NDCG@k** cuando tus documentos tienen grados de relevancia, por ejemplo `1`, `2` y `3`.

En general, un buen sistema debe recuperar documentos relevantes en los primeros puestos. Por eso las métricas `@1`, `@3`, `@5` y `@10` permiten analizar la calidad del ranking desde distintos niveles de exigencia.
