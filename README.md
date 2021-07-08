# Máquinas de Vectores de Soporte (SVM)
## Implementación en Python a partir de su fundamentación teórica
### Javier Guzmán Muñoz
#### Trabajo de Fin de Grado en Matemáticas: _Aprendizaje Automático: Marco PAC y Complejidad de Rademacher_

Este reporsitorio contiene la implementación de varias versiones del algoritmo de las máquinas de vectores de soporte basadas en la fundamnetación teórica de las mismas que define el algoritmo como aquel que da solución a un problema de programación cuadrática. Se emplea la biblioteca [_cvxopt_](https://cvxopt.org/) para resolver problemas de programación cuadrática. Las diferentes versiones del algoritmo que se inluyen son:
- Problema primal en el caso general (no separable).
- Problema dual en el caso general (no separable).
- Problema dual con _kernels_:
  + _Kernel_ polinómico.
  + _Kernel_ Gaussiano.
  + _Kernel_ sigmoide.
- Implementación de [_scikit-learn_](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- Implementación del problema primal en el caso separable.

Se trabaja también con varios conjuntos de datos, obtenidos de diversas fuentes y preprocesados epecñificamente para esta implementación. En la memoria de este trabajo pueden consultarse los detalles y la porcedencia de cada uno de ellos.

Además, se implementa el algoritmo de selección de hiperparámetros mediante validación cruzada.

Para medir el error del algoritmo se incluyen también dos mediadas: el error empírico (función de pérdida cero-uno) y la pérdida empírica con margen.

Para usar el programa hay que ejecutar el _script_ [`svm_main.py`](https://github.com/javigm98/SVM/blob/main/svm_main.py) y nos aparecerá un menú con todas las opciones que podemos elegir en cada momento. Basta navegar por las opciones para acceder a toda la funcionalidad del programa. Cuando trabajemos con alguno de los conjuntos de datos con dos características, se nos generará una imagen con la representación gráfica de la solución del algoritmo (salvo si usamos _kernels_), que se guardará en la carpeta [`Graphs`](https://github.com/javigm98/SVM/tree/main/Graphs). Esta visualización se lleva a cabo con las utilidades de la biblioteca [_matplotlib_](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html).
