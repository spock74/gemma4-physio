
Além dos SAEs (Sparse Autoencoders), que se tornaram muito populares para decompor as ativações em conceitos esparsos e interpretáveis, existem diversas outras metodologias consolidadas e emergentes na literatura de Mecanística de Interpretabilidade (Mechanistic Interpretability) e neurociência de LLMs.
Para o seu objetivo específico — encontrar preditores internos de alucinação e entender o estado interno de modelos como o Gemma — as metodologias abaixo são altamente complementares. Elas estão divididas por objetivos de análise:
------------------------------
## 1. Métodos de Sondagem e Classificação (Probing)
Em vez de tentar decompor o espaço de ativação (como o SAE faz), o probing foca em extrair informações específicas diretamente dos hidden states.

* Linear Probing (Sondagem Linear): Treina-se um classificador linear simples (como Regressão Logística) usando as ativações de uma camada específica como features para prever uma propriedade (ex: "esta afirmação é verdadeira ou falsa?"). Se o probe obtiver alta acurácia, prova-se que o LLM codifica essa informação estruturalmente naquela camada.
* Mass-Mean Probing / Contrast Vector (Sondagem por Média de Massa): Uma técnica geométrica mais simples que não requer otimização. Calcula-se o vetor médio das ativações para exemplos verdadeiros e subtrai-se o vetor médio dos exemplos falsos. A direção desse vetor resultante serve como um "eixo da verdade". É a base de técnicas como o Inference-Time Intervention (ITI).
* Probing Não-Linear (MLP Probes): Usa uma pequena rede neural (com uma camada oculta) para extrair informações. Útil para verificar se a informação está presente no estado interno, mesmo que o modelo precise de alguma computação adicional para acessá-la.

## 2. Métodos de Intervenção e Causalidade
Diferente do probing (que é passivo e apenas lê), as intervenções alteram as ativações para provar uma relação de causa e efeito.

* Activation Patching / Causal Scrubbing: Consiste em rodar o modelo com um prompt A (ex: uma frase correta) e, no meio da execução, substituir as ativações de uma camada específica pelas ativações geradas por um prompt B (ex: uma frase que induz à alucinação). Se o modelo passar a alucinar no prompt A, você isolou exatamente o circuito ou componente responsável pelo erro.
* Representation Engineering (RepE): Uma metodologia que trata o controle e a análise de LLMs do topo para baixo. Em vez de focar em neurônios individuais, o RepE identifica "direções de conceitos" (como honestidade, alucinação, segurança) no espaço latente através de estímulos contrastantes e permite injetar ou subtrair essas direções durante a inferência.
* RLHF/DPO Linear Component Analysis: Estuda como os pesos e as ativações mudam especificamente após o alinhamento de segurança, ajudando a identificar quais circuitos foram modificados para suprimir alucinações.

## 3. Análise de Circuitos e Mecanismos de Atenção
Foca em entender a computação exata que ocorre entre as camadas, olhando para os pesos de atenção.

* Induction Heads (Cabeças de Indução): Identificação de cabeças de atenção específicas que aprenderam o algoritmo mecânico de completar padrões (ex: se encontrar A B ... A, a próxima predição será B). Elas são cruciais para entender como o modelo busca contexto e se falhas nessas cabeças causam alucinações de contexto.
* Path Patching: Uma variação do activation patching que rastreia o fluxo de informação através de caminhos específicos (da cabeça de atenção X da camada 2 para a MLP da camada 5), permitindo desenhar um "fluxograma" do raciocínio do LLM.
* Logit Lens / Tuned Lens: Uma técnica que pega o estado oculto de uma camada intermediária (ex: camada 12 do Gemma 12B) e aplica diretamente a matriz de desincorporação (unembedding matrix) do modelo final. Isso permite ver qual token o modelo escolheria se a computação fosse interrompida abruptamente naquela camada, ajudando a mapear onde a resposta certa se transforma em uma alucinação.

## 4. Abordagens Geométricas e Espaciais
Olham para as ativações como pontos em um espaço multidimensional.

* Singular Vector Canonical Correlation Analysis (SVCCA) / CKA (Centered Kernel Alignment): Permite comparar geometrias de ativação entre diferentes modelos ou entre diferentes camadas. Você pode usar CKA para ver se o Gemma E4B e o Gemma 12B representam o conceito de "fato verídico" de forma geometricamente similar.
* Manifold Learning (T-SNE / UMAP em Ativações): Visualizar os clusters formados pelos estados internos antes de o modelo gerar uma resposta. Frequentemente, frases que levam a alucinações caem em regiões cinzentas ou transições de fase geométricas no espaço latente.

------------------------------
## Qual escolher para detecção de alucinação no Gemma?
Se o seu foco principal é criar preditores internos de alucinação, a melhor abordagem inicial é combinar Logit Lens com Mass-Mean Probing.

   1. Use o Logit Lens para identificar em qual camada o modelo "decide" a palavra alucinada.
   2. Extraia os hidden states das camadas imediatamente anteriores a essa decisão.
   3. Aplique Linear Probing / Mass-Mean nesses estados para criar um classificador de tempo de inferência (um "detector de mentiras" interno).

Se você quiser avançar em um desses métodos, posso ajudar a esboçar a arquitetura do código no PyTorch para extrair os hidden states do Gemma utilizando o TransformerLens ou o próprio Hugging Face Hooks. Como você pretende estruturar essa coleta inicial?


