"""
El problema del jugador pero como un problema de aprendizaje por refuerzo

"""

from RL import MDPsim, SARSA, Q_learning
from random import random, randint

class Jugador(MDPsim):
    """
    Clase que representa un MDP para el problema del jugador.
    
    El jugador tiene un capital inicial y el objetivo es llegar a un capital
    objetivo o quedarse sin dinero.
    
    """
    def __init__(self, meta, ph, gama):
        self.estados = tuple(range(meta + 1))
        self.meta = meta
        self.ph = ph
        self.gama = gama
        
    def estado_inicial(self):
        return randint(1, self.meta - 1)
    
    def acciones_legales(self, s):
        if s == 0 or s == self.meta:
            return []
        return range(1, min(s, self.meta - s) + 1)
    
    def recompensa(self, s, a, s_):
        return 1 if s_ == self.meta else 0
    
    def transicion(self, s, a):
        return s + a if random() < self.ph else s - a
    
    def es_terminal(self, s):
        return s == 0 or s == self.meta
    
mdp_sim = Jugador(
    meta=100, ph=0.40, gama=1
)

Q_sarsa = SARSA(
    mdp_sim, 
    alfa=0.2, epsilon=0.02, n_ep=300_000, n_iter=50
)
pi_s = {s: max(
    mdp_sim.acciones_legales(s), key=lambda a: Q_sarsa[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

Q_ql = Q_learning(
    mdp_sim, 
    alfa=0.2, epsilon=0.02, n_ep=300_000, n_iter=50
)
pi_q = {s: max(
    mdp_sim.acciones_legales(s), key=lambda a: Q_ql[(s, a)]
) for s in mdp_sim.estados if not mdp_sim.es_terminal(s)}

print("Estado".center(10) + '|' +  "SARSA".center(10) + '|' + "Q-learning".center(10))
print("-"*10 + '|' + "-"*10 + '|' + "-"*10)
for s in mdp_sim.estados:
    if not mdp_sim.es_terminal(s):
        print(str(s).center(10) + '|' 
              + str(pi_s[s]).center(10) + '|' 
              + str(pi_q[s]).center(10))
print("-"*10 + '|' + "-"*10 + '|' + "-"*10)

""" 
***************************************************************************************
Responde las siguientes preguntas:
***************************************************************************************
1. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?
2. ¿Para que sirve usar una politica epsilon-greedy?
3. ¿Qué pasa con la política óptima y porqué si p_h es mayor a 0.5?
4. ¿Y si es 0.5?
5. ¿Y si es menor a 0.5?
6. ¿Qué pasa si se modifica el valor de la tasa de aprendizaje?
7. ¿Qué pasa si se modifica el valor de gama?

***************************************************************************************

"""


""" 
***************************************************************************************
Respuestas:
***************************************************************************************
1. ¿Qué pasa si se modifica el valor de epsilón de la política epsilon-greedy?
   Al modificar el valor de epsilon en la política epsilon-greedy:
   
   - Epsilon alto (0.1-0.5): Mayor exploración. El agente prueba más acciones aleatorias, lo que puede ayudar a descubrir 
     estrategias óptimas que de otra manera podrían pasarse por alto. Sin embargo, la política final será más ruidosa y 
     podría no converger completamente a la óptima.
     
   - Epsilon bajo (0.01-0.05): Menor exploración. El agente explota más lo que ya ha aprendido, lo que puede llevar a una 
     convergencia más rápida si ya está cerca de la política óptima. Sin embargo, puede quedarse atrapado en óptimos locales.
     
   - Epsilon muy bajo o decreciente (0.001 o menos): Poca exploración. Útil en las etapas finales del aprendizaje para 
     "afinar" la política, pero puede limitar severamente la capacidad del agente para encontrar la política óptima si 
     se usa desde el principio.
   
   En el problema del jugador, un buen enfoque es comenzar con un epsilon relativamente alto (0.1) y disminuirlo gradualmente 
   a medida que avanza el aprendizaje, para balancear exploración y explotación.

2. ¿Para que sirve usar una politica epsilon-greedy?
   La política epsilon-greedy sirve para balancear la exploración y la explotación en el aprendizaje por refuerzo:
   
   - Explotación (1-epsilon): La mayor parte del tiempo (con probabilidad 1-epsilon), el agente elige la acción que 
     actualmente cree que es la mejor según sus estimaciones de Q. Esto le permite aprovechar lo que ya ha aprendido.
     
   - Exploración (epsilon): Una pequeña fracción del tiempo (con probabilidad epsilon), el agente elige una acción 
     aleatoria. Esto le permite descubrir nuevas estrategias que podrían ser mejores que las que ya conoce.
   
   Sin exploración (epsilon=0), el agente podría quedarse atrapado en una política subóptima porque nunca probaría 
   acciones alternativas que podrían llevar a mejores resultados. Sin explotación (epsilon=1), el agente nunca 
   aprovecharía lo que ha aprendido y su comportamiento sería completamente aleatorio.
   
   La política epsilon-greedy es una forma simple pero efectiva de implementar este balance, aunque existen otras 
   estrategias más sofisticadas como UCB (Upper Confidence Bound) o Thompson Sampling.

3. ¿Qué pasa con la política óptima y porqué si p_h es mayor a 0.5?
   Cuando p_h (probabilidad de ganar) es mayor a 0.5:
   
   La política óptima tiende a ser más agresiva, apostando cantidades mayores. Esto se debe a que el valor esperado 
   de cada apuesta es positivo:
   
   E[apuesta de a] = p_h * a - (1-p_h) * a = (2*p_h - 1) * a
   
   Cuando p_h > 0.5, este valor esperado es positivo, lo que significa que en promedio, apostar más te da más ganancias.
   
   En el límite, cuando p_h se acerca a 1, la política óptima se acerca a apostar todo lo posible en cada estado, 
   ya que casi siempre ganarás. Sin embargo, incluso con p_h > 0.5, la política óptima no siempre es apostar el máximo, 
   porque también hay que considerar el riesgo de perderlo todo antes de llegar a la meta.

4. ¿Y si es 0.5?
   Cuando p_h = 0.5 (probabilidad justa):
   
   El valor esperado de cualquier apuesta es cero:
   E[apuesta de a] = 0.5 * a - 0.5 * a = 0
   
   Esto significa que, en promedio, no ganas ni pierdes dinero con cada apuesta individual.
   
   En este caso, la política óptima es más compleja y depende de la estructura del problema. Para el problema del jugador 
   con p_h = 0.5, la política óptima tiende a ser apostar la cantidad mínima necesaria para poder llegar a la meta en un 
   solo golpe de suerte.
   
   Por ejemplo, si estás en el estado s=25 y la meta es 100, la política óptima sería apostar 25, porque si ganas, 
   llegarías a 50, y desde ahí podrías apostar 50 para llegar a la meta en la siguiente apuesta. Apostar más sería 
   innecesariamente arriesgado, y apostar menos te obligaría a hacer más apuestas para llegar a la meta.
   
   Esta estrategia se conoce como "apostar para llegar a una potencia de 2" o "estrategia de duplicación".

5. ¿Y si es menor a 0.5?
   Cuando p_h < 0.5 (probabilidad desfavorable):
   
   El valor esperado de cada apuesta es negativo:
   E[apuesta de a] = p_h * a - (1-p_h) * a = (2*p_h - 1) * a < 0
   
   Esto significa que, en promedio, pierdes dinero con cada apuesta.
   
   En este caso, la política óptima tiende a ser más conservadora, apostando cantidades menores para minimizar las pérdidas 
   esperadas. Sin embargo, como el objetivo es llegar a la meta (no maximizar el valor esperado), la política óptima no es 
   simplemente "no apostar".
   
   La estrategia óptima suele ser apostar lo mínimo posible en la mayoría de los estados, pero hacer apuestas más grandes 
   en ciertos estados estratégicos donde una victoria te pondría en una posición muy favorable.
   
   En el límite, cuando p_h se acerca a 0, la política óptima se acerca a hacer una sola apuesta grande (todo lo que tienes) 
   para intentar llegar a la meta de una vez, ya que múltiples apuestas pequeñas casi garantizan la pérdida total.

6. ¿Qué pasa si se modifica el valor de la tasa de aprendizaje?
   Al modificar el valor de alfa (tasa de aprendizaje):
   
   - Alfa alto (0.5-1.0): Aprendizaje rápido pero inestable. El agente actualiza sus estimaciones de Q rápidamente, 
     lo que puede acelerar el aprendizaje inicial, pero también puede causar oscilaciones y dificultades para converger 
     a la política óptima.
     
   - Alfa medio (0.1-0.3): Balance entre velocidad y estabilidad. El agente aprende a un ritmo razonable mientras 
     mantiene cierta estabilidad en sus estimaciones.
     
   - Alfa bajo (0.01-0.05): Aprendizaje lento pero estable. El agente actualiza sus estimaciones gradualmente, 
     lo que puede llevar a una convergencia más confiable pero requiere muchos más episodios.
   
   En el problema del jugador, un alfa de 0.2 (como en el código) es un buen compromiso. Para problemas más complejos 
   o ruidosos, podría ser beneficioso usar un alfa más bajo o incluso un alfa que disminuya con el tiempo.
   
   También es importante notar que la tasa de aprendizaje óptima está relacionada con otros parámetros como epsilon 
   y el número de episodios. Un alfa alto requiere más exploración (epsilon más alto) para evitar óptimos locales, 
   mientras que un alfa bajo requiere más episodios para converger.

7. ¿Qué pasa si se modifica el valor de gama?
   Al modificar el valor de gama (factor de descuento):
   
   - Gama bajo (0.1-0.5): El agente valora principalmente las recompensas inmediatas y descuenta fuertemente las futuras. 
     En el problema del jugador, esto llevaría a una política más miope que podría no ser óptima a largo plazo.
     
   - Gama medio (0.6-0.9): El agente balancea las recompensas inmediatas y futuras. Podría llevar a políticas que 
     toman algunos riesgos calculados.
     
   - Gama alto (0.9-0.999): El agente valora casi igualmente las recompensas futuras y las inmediatas. En el problema 
     del jugador, esto es apropiado porque el objetivo es maximizar la probabilidad de llegar a la meta eventualmente.
     
   - Gama = 1 (sin descuento): El agente valora todas las recompensas por igual, independientemente de cuándo ocurran. 
     Esto es apropiado para problemas episódicos con un objetivo claro como el problema del jugador.
   
   En el código, se usa gama=1, lo cual es adecuado para este problema episódico donde no hay preferencia temporal 
   (llegar a la meta en 10 pasos o en 100 pasos da la misma recompensa). Sin embargo, en problemas donde hay un 
   componente de tiempo o donde se quiere favorecer soluciones más rápidas, un gama menor a 1 sería más apropiado.
"""