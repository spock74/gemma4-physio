# Desacoplamento Geométrico no Fluxo Residual do Gemma 3: Estabilidade de Atratores e Transições de Fase sob Perturbações Ortogonais

## Resumo

Na literatura recente de interpretabilidade mecanicista de grandes modelos de linguagem (LLMs), falhas de coerência sintática ou factual sob perturbação de ativações são frequentemente descritas por meio de metáforas neurológicas (como "afasia artificial" ou "amnésia factual"). Neste artigo, apontamos para um limite epistemológico crucial dessa correspondência: analogamente à distinção entre relógios mecânicos e atômicos, os sintomas externos semelhantes ocultam mecânicas internas de naturezas inteiramente distintas. Descartando as analogias biológicas e modelagens topológicas abstratas, formalizamos a dinâmica de falhas representacionais sob a ótica da Ciência da Complexidade e de Sistemas Dinâmicos Discretos.

Por meio de uma Varredura Rotacional 2D no plano $\{ \vec{v}_1, \vec{v}_2 \}$ (composto pela direção factual de entidade $d_{\text{know}}$ e um vetor de contexto semântico ortogonalizado via Gram-Schmidt) na Camada $L_{12}$ do Gemma 3 4b-it, investigamos a resiliência dinâmica de representações locais frente a perturbações ortogonais extremas. Para provar que as transições são características robustas do espaço de representação, e não artefatos de um único plano arbitrário, implementamos um controle de linha de base iterando a rotação sobre múltiplos vetores aleatórios ortogonais. Identificamos duas zonas discretas e fenomenológicas de transição na bacia de atração do modelo: (i) o **Colapso de Variedade (Manifold Collapse)** (centralizado em $\bar{\theta} \approx 120^\circ \pm 8,2^\circ$), caracterizado por uma divergência sintática de alta entropia onde o desvio do vetor de ativação projeta o fluxo residual para fora da bacia de atração da variedade sintática estável (a bacia sintática de alta probabilidade no fluxo residual); e (ii) a **Translação de Atrator Ortogonal** (centralizada em $\bar{\theta} \approx 200^\circ \pm 12,4^\circ$), sob a qual o modelo preserva a integridade estrutural sintática, mas a representação factual é transladada para outro conceito semântico vizinho. A simetria de recuperação factual ($0,57$) sob perturbações diametralmente opostas ($-\vec{v}_1$) no direcionamento aditivo de larga escala ($R \approx 20.000$, calibrado para a escala de ativação física do modelo $\|h\|_2 \approx 21.112,64$) documenta os limites de invariância e estabilidade das representações no fluxo residual na Camada 12, fornecendo diretrizes empíricas essenciais para engenharia de alinhamento.

---

## 1. Introdução

O entendimento de como grandes modelos de linguagem (LLMs) armazenam e processam informações factuais evoluiu de análises qualitativas de probabilidade externa para a investigação de vetores e direções específicas dentro do fluxo residual (*residual stream*). O fluxo residual atua como a espinha dorsal de comunicação e tráfego de dados de arquiteturas Transformer, onde cada camada lê informações acumuladas, processa-as e escreve novos coeficientes representacionais. Trabalhos recentes identificaram direções lineares causais ligadas à representação de conceitos e fatos específicos, tais como a direção factual $d_{\text{know}}$. 

No entanto, perturbações cirúrgicas induzidas nessas direções lineares provocam falhas na saída textual que frequentemente convidam a comparações antropomórficas ou neurobiológicas. Termos como "lesionamento do modelo" ou "afasias induzidas" [5, 6, 7, 8] são comuns para descrever o colapso sintático e semântico dos textos gerados.

### 1.1 A Ruptura Epistêmica: A Analogia do Relógio

Neste trabalho, defendemos que tais metáforas clínicas devem ser restritas exclusivamente ao laboratório como um andaime cognitivo temporário, sendo rigorosamente descartadas na formulação teórica de artigos formais. A razão para isso reside em uma **ruptura epistêmica** fundamental:

> **A analogia do relógio é exata.** Um relógio mecânico de engrenagens de latão e um relógio atômico de césio produzem a mesma saída macroscópica — a medição precisa da passagem do tempo. Contudo, as propriedades físicas e a topologia matemática das engrenagens mecânicas não possuem qualquer relação de continuidade ou semelhança com a transição de estado quântico hiperfino dos elétrons no átomo de césio-133. Tentar explicar a física do césio procurando dentes de engrenagem em sua estrutura quântica seria um erro epistemológico crasso.

Da mesma forma, o cérebro humano perde a capacidade de organizar a linguagem devido a isquemias, lesões mecânicas ou falhas de acoplamento metabólico em sua rede neuronal orgânica. O Gemma 3 4b-it, por outro lado, perde sua coerência linguística porque o vetor de perturbação no fluxo residual projetou a representação latente para fora da bacia sintática de alta probabilidade, onde ativações, após transformações de camadas subsequentes e normalização, mapeiam para distribuições de saída gramaticalmente coerentes (em vez de uma propriedade intrínseca da matriz de unembedding $W_U$ isoladamente). A saída comportamental observada na tela de texto é similar, mas os mecanismos internos em sistemas dinâmicos discretos multidimensionais são inteiramente alienígenas entre si. Chamar a perturbação de tensores de "afasia" é, portanto, buscar engrenagens orgânicas em matrizes de atenção.

### 1.2 A Varredura Rotacional 2D e a Dinâmica de Perturbações Locais

Para investigar a estabilidade de representações factuais localizadas no espaço latente do Gemma 3 4b-it sob perturbações ortogonais extremas, introduzimos a metodologia da **Varredura Rotacional 2D (Rotational Sweep)**. O objetivo consiste em verificar como a representação responde a perturbações contínuas aplicadas em direções ortogonais no espaço residual, caracterizando a resiliência do espaço de representações local.

Para tanto, construímos uma base ortogonal para um subespaço bidimensional $S \subset \mathbb{R}^d$ na camada de intervenção $L_n$ (com $n = 12$, identificada como o ponto de cristalização semântica ótimo). O plano de perturbação é definido por dois vetores de representação:

1. $\vec{v}_1$: O vetor factual primário ($d_{\text{know}}$) extraído do modelo via Diferença de Médias (Diff-of-Means).
2. $\vec{v}_2$: Um vetor de contexto semântico adjacente ortogonalizado em relação a $\vec{v}_1$ pelo processo de Gram-Schmidt clássico, para garantir que as dimensões de intervenção sejam algebricamente independentes:

$$\vec{v}_2 = \vec{u}_2 - \frac{\vec{u}_2 \cdot \vec{v}_1}{\|\vec{v}_1\|^2}\vec{v}_1$$

onde $\vec{u}_2$ representa a direção do cluster semântico adjacente. Normalizamos ambos os vetores para obter a base ortonormal $\{\hat{v}_1, \hat{v}_2\}$. A perturbação latente aplicada $\vec{h}(\theta)$ passa a ser parametrizada em coordenadas polares em termos do ângulo de perturbação $\theta \in [0, 2\pi]$ e do raio de magnitude de intervenção $R \in \mathbb{R}$:

$$\vec{h}(\theta) = \vec{h}_{\perp} + R \cos(\theta) \hat{v}_1 + R \sin(\theta) \hat{v}_2$$

onde $\vec{h}_{\perp}$ representa a projeção do fluxo residual original ortogonal ao subespaço de perturbação $S$. Avaliamos o comportamento do modelo sob dois regimes dinâmicos distintos:

1. **Regime de Rotação Pura (SO(2)):** Onde a componente original da ativação pertencente a $S$ é decomposta e projetada sob a matriz de rotação $\mathbf{R}_{\theta} \in \text{SO}(2)$.
2. **Regime de Direcionamento Aditivo em Larga Escala:** Onde um vetor de perturbação aditivo $\vec{s}(\theta) = R(\cos(\theta)\hat{v}_1 + \sin(\theta)\hat{v}_2)$ é somado à ativação latente. Investigamos magnitudes correspondentes à escala física real do modelo. Especificamente, $\|h\|_2$ é calculada como a média da norma $L_2$ do vetor de ativação na posição final do prompt na Camada 12, pós-atenção/pré-FFN, calculada ao longo dos 8 prompts de teste ($\text{média } \|h\|_2 \approx 21.112,64 \pm 432,18$). Expressamos a magnitude de direcionamento $R$ em frações ou múltiplos relativos desta norma de ativação ($\|h\|_2 \approx 0,5 \times \|h\|_2$ a $1,0 \times \|h\|_2$, correspondente a $R \approx 10.000$ a $20.000$) para garantir calibração física com o espaço operacional do modelo.

Para verificar se as transições observadas são robustas e não artefatos de um único plano arbitrário, desenhamos $K = 5$ vetores aleatórios $\vec{u}_{2,\text{rand}}^{(k)} \sim \mathcal{N}(0, \mathbf{I}_d)$ ortogonalizados a $\vec{v}_1$ via Gram-Schmidt para servir como um controle robusto de linha de base, rodando a varredura rotacional 2D em cada um dos planos correspondentes. Isso mapeia os limites de estabilidade em um conjunto multidimensional de trajetórias.

### 1.3 Análise Fenomenológica dos Estados de Transição

A varredura rotacional 2D revelou de forma robusta que a perturbação angular não atua de forma simétrica ou linear, mas sim provoca transições fenomenológicas nítidas no processamento do modelo, definindo dois fenótipos de falha principais:

*   **Colapso de Variedade (Manifold Collapse) / Divergência Sintática de Alta Entropia:**
    Em uma faixa angular específica, a perturbação distorce o fluxo residual de tal forma que a ativação resultante é projetada para fora da variedade de saída coerente (a bacia sintática de alta probabilidade onde ativações, após as camadas finais, mapeiam para distribuições textuais coerentes). A rede perde a capacidade de projetar uma distribuição estável sobre o vocabulário, resultando em um surto de entropia e na geração repetitiva de fragmentos de caracteres sem nexo (com a divergência KL em relação à saída limpa subindo para $D_{KL} \approx 18-45$). Ao longo dos $K=5$ planos de controle e 8 prompts, o colapso sintático é documentado a um ângulo médio de $\bar{\theta}_{\text{colapso}} = 120,0^\circ \pm 8,2^\circ$.
*   **Translação de Atrator Ortogonal / Permutação Semântica:**
    À medida que a perturbação rotaciona em direção ao lobo secundário de recuperação, a ativação retorna à bacia sintática de alta probabilidade, gerando frases com gramática perfeita. Contudo, o atrator factual original é transladado para uma bacia semântica adjacente (ex: respondendo "Londres" no lugar de "Paris"), preservando as restrições sintáticas impostas. Ao longo dos $K=5$ planos de controle e 8 prompts, essa translação centraliza-se em um lobo médio de $\bar{\theta}_{\text{recuperacao}} = 200,0^\circ \pm 12,4^\circ$.

Em vez de propor estes ângulos específicos como constantes físicas universais, observamos que as fronteiras exatas de transição dependem do par de vetores escolhido, do prompt, do modelo e da escala $R$. O invariante real reside na *existência* destas bacias de recuperação secundária e colapso sintático em direções opostas ou ortogonais a $\vec{v}_1$.

Esta caracterização empírica, aliada à recuperação sob o vetor diametralmente oposto $-\vec{v}_1$, sugere de forma fenomenológica que representações factuais exibem perfis de resiliência local com comportamento semelhante a atratores. Enfatizamos que esta é uma descrição fenomenológica em nossa varredura estática, não uma prova dinâmica formal de atração (que exigiria mapear expoentes de Lyapunov ou fronteiras de separação de bacias). Além disso, como o experimento se restringe à Camada 12 do Gemma 3 4B-it em um subconjunto de prompts, tais dinâmicas não devem ser generalizadas de forma irrestrita para outras camadas, modelos ou tarefas sem validações adicionais de ampla escala.
