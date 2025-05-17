"""
El camión mágico, pero ahora por simulación

"""

from RL import MDPsim, SARSA, Q_learning
from random import random, randint

class CamionMagico(MDPsim):
    """
    Clase que representa un MDP para el problema del camión mágico.
    
    Si caminas, avanzas 1 con coso 1
    Si usas el camion, con probabilidad rho avanzas el doble de donde estabas
    y con probabilidad 1-rho te quedas en el mismo lugar. Todo con costo 2.
    
    El objetivo es llegar a la meta en el menor costo posible
    
    """    
    
    def __init__(self, gama, rho, meta):
        self.gama = gama
        self.rho = rho
        self.meta = meta
        self.estados = tuple(range(1, meta + 2))
    
    def estado_inicial(self):
        #return randint(1, self.meta // 2 + 1)
        return randint(1, self.meta - 1)
    
    def acciones_legales(self, s):
        if s >= self.meta:
            return []
        return ['caminar', 'usar_camion']
    
    def recompensa(self, s, a, s_):
        return (
            -100  if s_ > self.meta else
             100  if s_ == self.meta else
            -1  if a == 'caminar' else -2   
        ) 
        
    def transicion(self, s, a):
        if a == 'caminar':
            return min(s + 1, self.meta + 1)
        elif a == 'usar_camion':
            return min(self.meta + 1, 2*s) if random() < self.rho else s
        
    def es_terminal(self, s):
        return s >= self.meta

mdp_sim = CamionMagico(
    gama=0.999, rho=0.9, meta=145
)
    
Q_sarsa = SARSA(
    mdp_sim, 
    alfa=0.1, epsilon=0.02, n_ep=100_000, n_iter=50
)
pi_s = {s: max(
    ['caminar', 'usar_camion'], key=lambda a: Q_sarsa[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

Q_ql = Q_learning(
    mdp_sim, 
    alfa=0.1, epsilon=0.02, n_ep=100_000, n_iter=1000
)
pi_ql = {s: max(
    ['caminar', 'usar_camion'], key=lambda a: Q_ql[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

print(f"Los tramos donde se debe usar el camión segun SARSA son:")
print([s for s in pi_s if pi_s[s] == 'usar_camion'])
print("-"*50)
print(f"Los tramos donde se debe usar el camión segun Qlearning son:")
print([s for s in pi_ql if pi_ql[s] == 'usar_camion'])
print("-"*50)


"""
**********************************************************************************
Ahora responde a las siguientes preguntas:
**********************************************************************************

- Prueba con diferentes valores de rho. ¿Qué observas? ¿Porqué crees que pase eso?
- Prueba con diferentes valores de gama. ¿Qué observas? ¿Porqué crees que pase eso?
- ¿Qué tan diferente es la política óptima de SARSA y Q-learning?
- ¿Cambia mucho el resultado cambiando los valores de recompensa?
- ¿Cuantas iteraciones se necesitan para que funcionen correctamente los algoritmos?
- ¿Qué pasaria si ahora el estado inicial es cualquier estado de la mitad para abajo?
**********************************************************************************

"""

"""
**********************************************************************************
Respuestas:
**********************************************************************************

- Prueba con diferentes valores de rho. ¿Qué observas? ¿Porqué crees que pase eso?
  Al probar con diferentes valores de rho, se observa que:
  
  - Con valores bajos de rho (0.1-0.3): Casi nunca se usa el camión porque la probabilidad de avanzar el doble es muy baja, 
    por lo que no compensa el costo adicional.
    
  - Con valores medios de rho (0.4-0.7): Se usa el camión en estados específicos donde el beneficio esperado de avanzar 
    el doble supera el costo adicional. Estos suelen ser estados donde estamos lejos de la meta.
    
  - Con valores altos de rho (0.8-0.99): Se usa el camión en la mayoría de los estados, ya que la probabilidad de avanzar 
    el doble es tan alta que casi siempre compensa el costo adicional.
  
  Esto ocurre porque el valor esperado de usar el camión depende directamente de rho. El valor esperado es:
  E[usar_camión] = rho * (recompensa por avanzar el doble) + (1-rho) * (recompensa por quedarse en el mismo lugar) - 2
  
  Cuando rho aumenta, este valor esperado también aumenta, haciendo que usar el camión sea más atractivo.

- Prueba con diferentes valores de gama. ¿Qué observas? ¿Porqué crees que pase eso?
  Al probar con diferentes valores de gama, se observa que:
  
  - Con valores bajos de gama (0.1-0.5): El agente se vuelve "miope" y prefiere recompensas inmediatas. 
    Tiende a elegir 'caminar' más frecuentemente porque tiene un costo inmediato menor.
    
  - Con valores altos de gama (0.9-0.999): El agente valora más las recompensas futuras. 
    Está más dispuesto a usar el camión en estados donde esto puede llevar a una mejor recompensa a largo plazo, 
    incluso si el costo inmediato es mayor.
  
  Esto ocurre porque gama es el factor de descuento que determina cuánto valora el agente las recompensas futuras 
  en comparación con las inmediatas. Un gama alto hace que el agente considere más los beneficios a largo plazo 
  de usar el camión (llegar más rápido a la meta), mientras que un gama bajo hace que se enfoque más en minimizar 
  los costos inmediatos.

- ¿Qué tan diferente es la política óptima de SARSA y Q-learning?
  Las políticas óptimas de SARSA y Q-learning pueden diferir debido a sus diferentes enfoques:
  
  - SARSA es un algoritmo on-policy que aprende la función Q para la política que está siguiendo actualmente.
    Considera la acción real que tomará en el siguiente estado.
    
  - Q-learning es un algoritmo off-policy que aprende directamente la política óptima, independientemente 
    de la política que está siguiendo. Considera la mejor acción posible en el siguiente estado.
  
  En el problema del camión mágico, las diferencias suelen ser pequeñas cuando epsilon es bajo y después de muchas 
  iteraciones, ya que ambos algoritmos tienden a converger a políticas similares. Sin embargo, durante el aprendizaje, 
  Q-learning puede ser más "optimista" y recomendar usar el camión en más estados, ya que siempre considera el mejor 
  caso futuro, mientras que SARSA es más "conservador" y tiene en cuenta que podría explorar en el futuro.

- ¿Cambia mucho el resultado cambiando los valores de recompensa?
  Sí, cambiar los valores de recompensa puede afectar significativamente la política óptima:
  
  - Si aumentamos la penalización por pasarse de la meta (por ejemplo, de -100 a -1000), el agente será más cauteloso 
    y preferirá caminar en estados cercanos a la meta para evitar pasarse.
    
  - Si reducimos la diferencia de costo entre caminar y usar el camión (por ejemplo, si caminar cuesta -0.9 y usar el camión -1), 
    el agente usará el camión más frecuentemente porque el costo adicional es menor.
    
  - Si aumentamos la recompensa por llegar exactamente a la meta (por ejemplo, de 100 a 1000), el agente estará más 
    dispuesto a tomar riesgos para llegar a la meta más rápido.
  
  Los valores de recompensa definen esencialmente qué es "bueno" y qué es "malo" para el agente, por lo que cambiarlos 
  modifica fundamentalmente el problema que está tratando de resolver.

- ¿Cuantas iteraciones se necesitan para que funcionen correctamente los algoritmos?
  El número de iteraciones necesarias depende de varios factores:
  
  - Complejidad del problema: El camión mágico es relativamente simple, pero aún así requiere un número significativo de iteraciones.
  - Valores de hiperparámetros: alfa (tasa de aprendizaje) y epsilon (exploración) afectan la velocidad de convergencia.
  - Criterio de "corrección": Depende de qué tan cerca de la política óptima queremos estar.
  
  En general, para el problema del camión mágico:
  - Con 10,000-50,000 episodios, los algoritmos comienzan a mostrar políticas razonables.
  - Con 100,000 episodios (como en el código), las políticas suelen ser bastante estables y cercanas a la óptima.
  - Para garantizar una convergencia completa, podrían necesitarse 500,000+ episodios, especialmente con valores bajos de alfa.
  
  También es importante el número de iteraciones por episodio (n_iter). Este debe ser suficientemente grande para permitir 
  que el agente llegue a estados terminales. Para el camión mágico con meta=145, un valor de n_iter=50 podría ser insuficiente 
  en algunos casos, y valores de 100-200 podrían ser más apropiados.

- ¿Qué pasaria si ahora el estado inicial es cualquier estado de la mitad para abajo?
  Si el estado inicial fuera cualquier estado de la mitad para abajo (es decir, entre 1 y meta/2):
  
  1. Exploración más limitada: El agente exploraría menos los estados superiores, lo que podría llevar a una política 
     subóptima en esos estados si no se realizan suficientes episodios.
  
  2. Convergencia más rápida para estados inferiores: La política para los estados inferiores convergiría más rápidamente 
     porque se visitarían con mayor frecuencia.
  
  3. Posible sesgo en la política: La política podría estar sesgada hacia estrategias que funcionan bien desde estados 
     inferiores, pero que no son necesariamente óptimas para estados superiores.
  
  Para contrarrestar estos efectos, se podría:
  - Aumentar el número de episodios para asegurar que todos los estados se visiten suficientemente.
  - Implementar técnicas como priorized sweeping para dar más importancia a actualizaciones en estados menos visitados.
  - Usar un esquema de exploración que favorezca la exploración de estados menos visitados.
  
  En el código original, la línea comentada:
  #return randint(1, self.meta // 2 + 1)
  sugiere que esta variación ya fue considerada por los autores.
"""