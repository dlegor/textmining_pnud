## Descripción y orden de uso de los Notebook

Para replicar los experimentos o hacer algunas variaciones se requiere seguir el siguiente orden en la ejecución de los Notebook( o scripts, en algunos casos).

1.- Limpieza de Base

El notebook de nombre **Limpieza_Base.ipynb**, hace un cambio en la organización de la información y limpia algunas de las variables presentes en el archivo: prog_avance_de_indicadores.csv

La salida de este notebook es un archivo en formato feather, esto para hacer más rápida su lectura.

2.- Procesamiento del PDF

El notebook de nombre **PDF_Textos.ipynb**, procesa y guarda la información sobre el PDF del archivo:Guía_Carga_avances_Módulo_CP_2019.pdf

Cabe mencionar que dicho notebook se puede emplear para procesar y guardar los textos del PDF de cualqueir otro año, lo único que se debe de hacer es cambiar en las celdas o campos los textos correspondientes al PDF que se desea procesar y cambiar el nombre en caso de querer contar con más de una versión de PDF.

3.- Selección de Muestra 

El notebook de nombre **Textos_Elegidos.ipynb** selecciona y organiza diferentes archos que son requeridos en el experiemento. El objetivo de este archivo es seleccionar bajo un mínimo de criterios los indicadores que pueden ser objeto de experimentar y explorar.

Dicha muestra permite generar tres archivos:
 * Muestra para estimar el indicador de Complejidad
 * Muestra de textos para estimar la similitud 
 * Submuestra de los indicadores con causas reportadas del año 2019

 4.- Procesamiento de los Textos
 El notebook de nombre **Textos_Procesados.ipynb**, este archivo procesa los textos seleccionados para el experimento y genera 3 versiones del texto original reportado. Para el experimento de hace uso del procesamiento por oraciones, pero puede ser remplazado por cualqueir de las otras dos versiones generadas.

 El arhivo que se obtiene de este procesamiento es guardado y requerido para los experimentos.


 *Para ver más detalles de las etapas procesadas se puede revisar en la documentación del notebook*

 4.- Estimación de Embeddings

 El notebook, de nombre **Cálculo_de_Embeddings.ipynb**, este archivo genera 3 archivos de salida, se tienen 3 opciones de modelos de lenguaje o de modelos para generar una representación vectorial de los textos.

 Los tres archivos de salida son los siguientes:
 * Representación vectorial de las oraciones
 * Represnetación vectorial de los textos del PDF
 * Representación vectorial de todo el texto sin importar la división por oraciones

## Lista de archivos

**Archivos de Entrada**

* prog_avance_de_indicadores.csv
* Causas Generales CP2019.xlsx
* Guía_Carga_avances_Módulo_CP_2019.pdf( solo el texto descargado en el Notebook PDF_Textos.ipynb)

**Archivos Generados**
* base
* base_txt
* base_casuas_2019   
* base_processed_txt 
* base_txt_prog
* Txt_Guia_Causas
* embedding_matrix_sentences_roberta.npy
* embedding_matrix_txt_roberta.npy
* embedding_matrix_causas_roberta.npy

## Estimación de Indicador de Complejidad y Matrices de Similitud

### Indicador de Complejidad

Para generar el indicador de complejidad se requeire el Notebook **Indice_Complejidad.ipynb** y la base **base_txt**. Las instrucciones de ejecución y una muestra de como se obtiene el resultado se pueden encontrar en el notebook.

Dependiendo de las condiciones de computo donde se ejecute, este puede necesitar en promedio  1 hrs y 30 min.


### Estimación de Similitudes

Para generar la matriz de similitudes y la tabla con la información de corte o de qué indicadores pasan los umbrales para cada Causa, se necesita el notebook **Estimacion_Similitud.ipynb** y los archivos:

* embedding_matrix_sentences_roberta.npy
* base_casuas_2019
* embedding_matrix_causas_roberta.npy

Estos tres archivos pueden cambiar debido a que se puede emplear otro modelo de lenguaje para generarlos.Al final de la ejecución se obtendran 2 tablas con la información de la similitud.

Dependiendo de las condiciones de computo, este puede requerir en promedio 1 hr y 45 min para 50 iteraciones. Lo recomendable son de 50 a 100 iteraciones.



