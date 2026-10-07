# PlagaScan — Dataset de entrenamiento (sano / pulgón amarillo en frijol)

Documento de decisiones técnicas para el módulo de dataset. Generado como parte
del pipeline de entrenamiento (transfer learning MobileNetV2, input 224x224x3).

**Estado: CERRADO.** Dataset final en `~/plagascan/dataset/{train,val}/{sano,plaga}/`.

---

## 1. Especie / criterio visual

**Especie objetivo: *Aphis gossypii* (pulgón del algodonero), morfo amarillo.**

Justificación:
- *Aphis craccivora* ("pulgón de las leguminosas") es la especie más citada en
  literatura fitosanitaria mexicana para frijol, pero su coloración típica es
  **negra/café oscura**. No coincide con el nombre de trabajo del proyecto
  ("pulgón amarillo").
- *Aphis gossypii* está confirmada como plaga de frijol (junto con algodón,
  melón, pepino, tomate) y su color es explícitamente variable: oscuro, verde
  oliva, verde claro **o amarillo**, según temperatura y densidad poblacional.
  Es el mejor match documentado para el criterio "amarillo".

**Criterio de etiquetado usado (no morfológico, apto para cámara de celular):**
coloración amarillenta / verde-amarillenta, formación en colonia densa en el
envés de la hoja o brotes tiernos.

**Nota para la defensa:** durante la curación del dataset se confirmó
visualmente que *Aphis gossypii* en campo aparece mayoritariamente en morfos
oscuros/negros/verdes — el morfo amarillo es una minoría de las observaciones
públicas disponibles (ver sección 3). Esto es consistente con la literatura
consultada y no es un error de la fuente de datos; es información real sobre
la variabilidad de la especie que se documenta explícitamente como limitación
del criterio de color como único diferenciador.

---

## 2. Fuentes de datos — resultado final

| Fuente | Clase | Licencia | Cantidad final usada |
|---|---|---|---|
| Makerere iBean (via Hugging Face `AI-Lab-Makerere/beans`) | `sano` | **MIT** | 200 |
| LeLePhid (Mendeley, DOI 10.17632/tndhs2zng4) | `plaga` | CC BY 4.0 | 150 |
| iNaturalist (observaciones *Aphis gossypii*, curadas a mano) | `plaga` | CC-BY / CC-BY-NC (por imagen, ver `inat_metadata.json`) | 24 |
| **Total** | | | **374** (200 sano / 174 plaga) |

Correcciones respecto a la versión inicial de este documento:
- La licencia real de Makerere/iBean es **MIT** (confirmado en el README del
  repo oficial), no CC BY 4.0 como se apuntó inicialmente por confundirla con
  la licencia de un paper que describe el dataset.
- El bucket original de Google Cloud Storage (`storage.googleapis.com/ibeans/*`)
  ya no es público (HTTP 403) — se usó el espejo en Hugging Face
  (`AI-Lab-Makerere/beans`), que sí mantiene el dataset descargable.
- LeLePhid requiere descarga manual desde Mendeley (bloquea descarga
  automatizada por código, HTTP 404 en el endpoint público).

### Proceso de curación de iNaturalist (importante para la defensa)

La API de iNaturalist, al buscar por nombre de especie (`Aphis gossypii`), no
permite filtrar por morfo de color — devuelve todas las observaciones de la
especie sin importar su coloración real (oscura, verde o amarilla). Se
intentó primero un filtro automático por umbral de tono en espacio HSV
(OpenCV), pero se descartó tras revisión visual: con un umbral permisivo
pasaban 139/150 imágenes (falso positivo por detectar el verde-amarillento
del fondo de hoja, no del insecto); con un umbral más estricto pasaban 34/150
pero solo 4 de esas 34 eran realmente amarillas al inspeccionarlas.

**Se optó por curación manual**: se generaron contact sheets numerados de las
150 imágenes descargadas y se revisaron una por una, confirmando **24
imágenes** con morfo amarillo/dorado claramente identificable. Los índices
curados están fijos en el código (`INAT_YELLOW_INDICES` en
`build_dataset.py`) y aislados por fecha de descarga para ser robustos a que
la carpeta local crezca en descargas posteriores.

**Limitación declarada:** 24 imágenes es una muestra pequeña del morfo
amarillo real. Es la cifra honesta disponible en fuentes con licencia abierta
al momento de construir el dataset — se documenta como candidato prioritario
para ampliar con fotografía de campo propia (fase ya planeada en el
documento U3_A1 del proyecto).

### Verificación visual de LeLePhid (importante para la defensa)

Se verificó a resolución completa (no solo miniatura) una muestra aleatoria
de 4 imágenes de la clase `aphid` de LeLePhid. Se confirmó que el criterio de
anotación original del dataset es la presencia de **pequeñas manchas
blancas** (cera/secreción) sobre la hoja de limón — un patrón de textura
sutil y distinto al patrón de color amarillo del insecto que es el criterio
central de PlagaScan. Por esta razón, se limitó su peso en el dataset final a
150 imágenes (de las 335 disponibles) para que no domine desproporcionadamente
sobre las 24 imágenes de iNaturalist, que sí capturan el patrón visual
correcto (insecto amarillo real).

**Conclusión metodológica:** LeLePhid aporta principalmente diversidad de
"textura foliar anómala + colonia de insectos pequeños", útil como
regularización para que el modelo no sobreajuste a un solo tipo de fondo,
pero **no debe interpretarse como fuente principal del patrón de color
amarillo** — ese rol lo cumple el subconjunto curado de iNaturalist.

---

## 3. Estructura final

```
~/plagascan/dataset/
├── train/
│   ├── sano/     (160 imágenes, fuente: Makerere/MIT)
│   └── plaga/    (139 imágenes: 150×0.8≈120 LeLePhid + 24×0.8≈19 iNaturalist, redondeo real via shuffle)
├── val/
│   ├── sano/     (40 imágenes)
│   └── plaga/    (35 imágenes)
├── raw/                    # descargas originales (respaldo, no usar directo para entrenar)
├── metadata.csv             # trazabilidad completa: archivo, clase, split, fuente, licencia, origen
├── resumen.md                # conteo generado automáticamente
├── ejemplos.png               # grid visual de muestra
└── inat_metadata.json          # licencia específica por observación de iNaturalist
```

- **Split:** 80% train / 20% val, aleatorio con semilla fija (`random.seed(42)`).
- **Formato:** `.jpg`, redimensionado a 224×224 con letterbox (padding negro).
- **Nomenclatura:** `{clase}_{fuente}_{índice:04d}.jpg`.
- **Balance final:** 200 sano / 174 plaga — razonablemente balanceado, ambas
  clases superan el mínimo objetivo de 150 imágenes reales fijado al inicio.

---

## 4. Limitaciones generales declaradas (resumen para la defensa)

1. **Mezcla de hospederos/especies**: Makerere es frijol (variedad ugandesa,
   no mexicana); LeLePhid es limón, no frijol. No existe a la fecha un
   dataset público de *Aphis gossypii* amarillo sobre hoja de frijol
   mexicano — es la razón de fondo para combinar fuentes.
2. **Muestra pequeña del patrón central** (morfo amarillo real): solo 24
   imágenes, curadas manualmente por ausencia de filtro automático confiable.
3. **LeLePhid aporta un patrón de textura distinto** (manchas blancas) al
   patrón de color que es el criterio de clasificación del proyecto — su
   peso en el dataset se limitó deliberadamente por esta razón.
4. El sistema sigue siendo, por diseño, una **alerta temprana** y no un
   diagnóstico entomológico certero — esto es consistente con el alcance ya
   declarado en la documentación de proyecto (U2_P3).

**Recomendación para la siguiente fase:** priorizar la captura de fotografía
propia de campo (ya planeada) específicamente del morfo amarillo sobre hoja
de frijol, para reducir la dependencia del subconjunto curado de 24 imágenes
y mejorar tanto el volumen como la fidelidad al contexto real del proyecto.

---

## 5. Ampliación de iNaturalist (segunda tanda)

Se amplió la búsqueda agregando *Aphis craccivora* como segunda especie
candidata y bajando `quality_grade` de solo `research` a incluir también
`needs_id`, buscando hasta 500 imágenes por especie (1000 en total).

**Resultado:** de 1000 imágenes buscadas, tras pre-filtro automático de color
(HSV, ajustado hasta min_ratio=0.25 para llegar a un número revisable a mano)
y curación visual manual, solo **7 imágenes nuevas** fueron confirmadas como
morfo amarillo genuino — una tasa de acierto de 0.7%, muy por debajo del 16%
de la tanda original (150 imágenes → 24 confirmadas).

## 6. Fuente descartada: dataset de "conchuela"

Se evaluó una fuente adicional (dataset público, 590 imágenes, carpeta
nombrada genéricamente "plaga") para ampliar la clase `plaga`. Tras
inspección de nomenclatura de archivos, se confirmó que el 100% de las
imágenes corresponden a **conchuela** (*Epilachna varivestis*), un
escarabajo defoliador del frijol — una plaga real y relevante para el
cultivo, pero **distinta de la plaga objetivo del proyecto** (pulgón
amarillo, *Aphis gossypii*).

**Decisión:** no se integra a la clase `plaga` actual, para no contaminar el
criterio de clasificación con una plaga visualmente muy distinta (escarabajo
vs. áfido). Se conserva sin usar en
`~/plagascan/dataset/raw/manual_source_1/` como activo para una futura fase
del proyecto, en caso de que se amplíe el alcance a múltiples plagas (el
planteamiento original del proyecto, antes de la corrección de alcance
solicitada por los sinodales, contemplaba justamente esta posibilidad).

## 7. Tercera fuente: manual_source_2 (Wikimedia + iNaturalist, herramienta propia)

Se incorporó una fuente adicional generada por el usuario con una herramienta
propia de scraping que agrega imágenes de Wikimedia Commons e iNaturalist
para el taxón *Aphis gossypii*, con metadata de licencia por imagen
(`metadata.csv`: columnas `source`, `license`, `creator`, `page_url`, etc.).

**Proceso de verificación y filtrado:**

1. **Filtro de licencia:** de 2000 filas totales, 344 tenían licencia vacía
   (excluidas) y 31 tenían licencia "ND" — *No Derivatives* (excluidas por
   precaución legal, ya que el estatus de entrenar modelos de IA con
   contenido "ND" es legalmente ambiguo). Quedaron **1625 imágenes usables
   por licencia**.
2. **Filtro de color automático** (mismo método HSV usado en tandas
   anteriores), ajustado progresivamente de min_ratio=0.10 (353 candidatas)
   a 0.20 (200) a 0.30 (126 candidatas, número manejable para revisión
   manual).
3. **Curación manual** de las 126 candidatas vía contact sheets numerados.

**Resultado: 7 imágenes confirmadas de 126 revisadas (de 2000 originales)** —
tasa de acierto ~0.4% sobre el total, consistente con el patrón de retorno
decreciente ya observado.

**Conclusión acumulada tras 3 intentos independientes de ampliar fuentes
públicas:**

| Tanda | Buscadas | Confirmadas | Tasa de acierto |
|---|---|---|---|
| Original (iNaturalist, *A. gossypii*) | 150 | 24 | 16% |
| Ampliación (iNaturalist, 2 especies) | 1000 | 7 | 0.7% |
| manual_source_2 (Wikimedia + iNaturalist) | 2000 (1625 con licencia válida) | 7 | ~0.4% |
| **Total acumulado** | | **38** | |

La tasa de acierto decreciente y consistente en 3 fuentes independientes
confirma que las fuentes públicas de fácil acceso para el morfo amarillo
específico están efectivamente agotadas. Se decidió cerrar la búsqueda de
más datos vía fuentes públicas en este punto — el tiempo de curación manual
adicional ya no se justifica frente al beneficio marginal esperado.

## 8. Dataset final consolidado

| Clase | Train (real) | Val (real, 100%) | Total real |
|---|---|---|---|
| sano | 160 | 40 | 200 |
| plaga | 150 | 38 | **188** |

**Composición final de la clase `plaga` (188 imágenes):**
- 150 LeLePhid (CC BY 4.0) — patrón de textura/colonia, peso limitado deliberadamente
- 24 iNaturalist, tanda original (CC-BY / CC-BY-NC según imagen)
- 7 iNaturalist, tanda de ampliación (mismas licencias)
- 7 Wikimedia/iNaturalist, manual_source_2 (licencias variadas, verificadas individualmente en `metadata.csv`)

**Composición de la clase `sano` (200 imágenes):**
- 200 Makerere iBean vía Hugging Face (MIT)

**Siguiente paso ejecutado:** data augmentation sobre `train/` únicamente
(rotación, espejeo horizontal, ajuste de brillo/contraste), vía
`augment_dataset.py`, multiplicador 4×. `val/` se mantiene 100% con imágenes
reales, sin aumentar, para preservar la validez de las métricas de
evaluación del modelo.

## 9. Totales finales para la fase de entrenamiento

| Split | sano | plaga | Total |
|---|---|---|---|
| **train** (con augmentation) | 640 | 600 | 1240 |
| **val** (100% real) | 40 | 38 | 78 |
| **Total efectivo** | 680 | 638 | **1318** |
