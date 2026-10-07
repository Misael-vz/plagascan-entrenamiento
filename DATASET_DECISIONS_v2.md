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
