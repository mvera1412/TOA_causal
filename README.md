# TOA_causal


### Crear datos:
Para crear los datos basta con correr: 
	python synthetic_data_generator.py ../datos_toa/

Obviamente se puede cambiar el directorio donde guardar los datos, pero recordar que no entran en github. Si quiero cambiar las incertezas en la posición se debe modificar las variables spe y spetest del archivo "~/TOA_causal/synthetic/createdata.py". Actualmente están seteadas como:
	spe=np.array([1.0, 0.1, 0.01, 0.001, 0.0001])
	spetest = np.array([0.0, 1.5])
	
### Correr desde Jupyter
Hay que ejecutar las celdas de "\~/TOA_causal/TUPAC_toa.ipynb". Guardará los modelos en la carpeta donde están los datos, un log con resultados de validación en "~/TOA_causal" y mostrará los resultados de test por pantalla. 

### Correr por consola
Hay que ejecutar el archivo main.py. Su comportamiento es simil a la versión jupyter. Observar las variables del encabezado, las mismas deberán ser ajustadas de acuerdo a las necesidades:
	1)val_percent: Porcentaje de datos (originalmente catalogados como train) que se reservarán para validación. Está seteado para que no haya redondeo.
	2)le: Cantidad de environments (train). Venimos trabajando con 5.
	3)epochs: Cantidad de epochs a correr. Esta implementación corre el algoritmo durante todos los epochs. El early stopping funciona a posteriori restaurando el mejor epoch. Actualmentecorrimos 50 pero está quedando corto. Se debería aumentar.
	4)cache_dir: Directorio donde están los datos.
	5)fecha: Es simplemente un indicativo para guardar los modelos y los resultados.
	6)alphas: Learning rates a validar. Actualmente la validación está definida en grilla. Más adelante habría que implementar una versión random.
	7)bs: Idem alphas pero con batchsize per environment. Para hacer justa la comparación entre la red causal y la bechmark, en ésta última se utilizará como batchsize le*bs.
	8)taus: Idem alphas pero con los agreement threshold. Actualmente están seteados en [0.4,0.8], valores típicos para 5 environment. Pensar que cada elemento del gradiente puede ser positivo o negativo (también cero, pero es mucho más infrecuente), con lo cuál con 5 enviroments el contraste puede salir 5 a 0, 4 a 1 o 3 a 2. Como los signos positivos pesan +1 y los negativos -1, el valor absoluto de la suma puede dar 5, 3 o 1. Lo que en promedio da 1.0, 0.6 y 0.2. Notar que 0.4 y 0.8 son los valores intermedios. En este sentido el agreement threshold 0.4 enmascará solamente los 3 a 2, mientras que el 0.8 enmascarará los 4 a 1 y los 3 a 2.
	
### Plotear resultados del log
El archivo "~/TOA_causal/plotter.py" genera unas curvas a partir del log que a mi me parecieron interesantes. Siempre ajustar la variable "file_name" al log que se desea graficar. 
La primera de las curvas compara las curvas de entrenamiento (epochs vs loss) del mejor resultado de la red causal con el mejor resultado de la red benchmark. Dado que el resultado es súper ruidoso (aunque menos ruidoso en la causal), para el resto de las curvas voy a comparar las envolventes convexas.
La segunda de las curvas compara el mejor resultado de cada red para cada batchsize. ¿Cuál es el objetivo? Si los mejores resultados son para batchsize altos habría que agregar más grandes en la validación (porque hay margen de mejora) y quizás eliminar los más pequeños (que son los que más tardan).
La tercera de las curvas compara el mejor resultado de cada red para cada learning rates. De nuevo, la idea es observar donde conviene intensificar la búsqueda.
La cuarta compara los mejores resultados para cada agreement threshold. Acá no hay mucho que ajustar pero estas curvas están buenas para extraer conclusiones.
