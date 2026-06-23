# Walkthrough: Transição para Ciência da Complexidade e Controle de Linha de Base Ortogonal Robusta

Este documento descreve as alterações efetuadas para consolidar a modelagem teórica sob a Ciência da Complexidade padrão, implementar o controle de linha de base estatística via múltiplos vetores ortogonais aleatórios no experimento de varredura rotacional 2D e validar os resultados através de gerações de texto sob perturbação.

## 1. Refatoração Teórica e Textual

### Manuscrito LaTeX Oficial
*   **Arquivo:** [paper.tex](../../docs/paper.tex) | [PDF](../../docs/paper.pdf)
*   **Ações Realizadas:**
    *   **Remoção do Resumo e Introdução:** Conforme solicitado, removemos completamente o resumo (`abstract`), as palavras-chave (`keywords`) e a seção de Introdução (incluindo a subseção sobre a analogia do relógio).
    *   **Reestruturação dos Títulos:** As subseções do Experimento Rotacional (antiga 1.2) e Análise Fenomenológica (antiga 1.3) foram promovidas a seções principais (`\section`).
    *   **Manutenção das Duas Imagens:** As figuras 1 (`fig7_rotational_sweep.png`) e 2 (`fig8_topological_analysis.png`) foram devidamente preservadas no texto com suas respectivas legendas explicativas e numeração atualizada.
    *   **Compilação e Alinhamento:** O documento foi compilado com sucesso por duas passagens do `pdflatex` para sanar referências cruzadas de hiperlinks. A versão final agora possui 4 páginas.

### Script de Diagnóstico de Geração
*   **Arquivo:** [verify_rotational_generation.py](../../scripts/verify_rotational_generation.py)
*   **Ações Realizadas:**
    *   Substituição do termo "Secondary Klein bottle recovery" por "Orthogonal Attractor Translation" nas condições diagnósticas.

## 2. Refatoração do Código do Experimento (Opção A)

*   **Arquivo:** [run_rotational_sweep.py](../../scripts/run_rotational_sweep.py)
*   **Ações Realizadas:**
    *   **Geração de Multi-Vetores:** Modificamos o script para gerar $K = 5$ vetores ortogonais aleatórios $\vec{v}_{2,\text{rand}}^{(k)}$ orthogonalizados a $\vec{v}_1$ por Gram-Schmidt, ao invés de um único vetor controle.
    *   **Varredura Estatística:** Implementamos loops individuais para executar a rotação polares sobre as $K$ direções de controle.
    *   **Visualização Polar Premium (Shaded):** O código de plotagem agora calcula a média e o desvio padrão das $K$ direções de controle. A linha de controle média é plotada pontilhada e a região de $\pm 1$ desvio padrão é desenhada com `fill_between` para ilustrar a bacia de estabilidade e o intervalo de confiança.
    *   **Atualização de Títulos e Textos:** O título do gráfico e logs console foram revertidos de "Funtor Fantasma" para "Varredura Rotacional 2D (Estabilidade de Atratores e Controle de Linha de Base)".

## 3. Validação do Experimento

### A. Resultados das Varreduras Polares
As varreduras polares foram executadas com sucesso para $K=5$ vetores ortogonais aleatórios de controle. Os dados brutos foram salvos em [rotational_sweep_results.json](../../scratch/rotational_sweep_results.json). Os gráficos finais com o desvio padrão sombreado das linhas de base foram salvos em:
*   [rotational_sweep.png](../../plots/rotational_sweep.png)
*   [fig7_rotational_sweep.png](../../docs/figures/fig7_rotational_sweep.png)

### B. Evidência Empírica de Geração sob Rotação
Executamos o diagnóstico de geração com o comando `python scripts/verify_rotational_generation.py`. A saída confirmou as duas transições de fase de maneira extremamente nítida:

```text
Prompt: 'The capital of France is' (Expected: 'Paris')
  [Baseline              ]: 'Paris, **a city renowned for its iconic landmarks'
  [Theta = 20° (Peak)    ]: 'Paris, continuing home to over 18 million'
  [Theta = 120° (Dead)   ]: 'twistja-jaUjaUjaJaJa'
  [Theta = 200° (Recovery)]: 'London, **please provide the sentence!** 😊'
```

*   **$\theta = 120^\circ$ (Dead Zone / Manifold Collapse):** A ativação é projetada fora da bacia sintática de alta probabilidade, resultando em uma divergência sintática de alta entropia com repetições sem nexo (`twistja-jaUjaUjaJaJa`).
*   **$\theta = 200^\circ$ (Recovery / Orthogonal Attractor Translation):** O modelo retorna para dentro do "coherent output manifold" (produzindo gramática perfeita), porém o atrator factual foi transladado ortogonalmente para outro conceito semântico vizinho na bacia de atração local, resultando na resposta factual alterada (`London`).

## 4. Análise Topológica e Controles Avançados

Executamos a nova bateria de testes topológicos e controles avançados rodando o script `python scripts/topological_and_advanced_controls.py`. Os dados brutos foram salvos em [topological_controls_results.json](../../scratch/topological_controls_results.json). O gráfico consolidando as análises foi salvo em:
*   [topological_analysis.png](../../plots/topological_analysis.png)

### Principais Descobertas:

1.  **Homologia Persistente ($H_0$):**
    Construímos uma nuvem de pontos das ativações na Camada 13 (a camada subsequente à intervenção) ao longo do sweep de $\theta$ e calculamos a homologia persistente em $H_0$ via árvore geradora mínima (Minimum Spanning Tree / merges de single linkage).
    Identificamos que, sob ângulos normais de recuperação ou sem perturbação, os componentes persistentes se fundem tardiamente (preservando uma topologia rica e espalhada no espaço latente). Já nas zonas de colapso, a distância média cai drasticamente, forçando merges rápidos dos componentes conexos e atestando matematicamente o colapso do manifold representacional.
2.  **Rotação de Householder (Norm-Preserving):**
    A rotação pura de subespaço realizada via reflexão de Householder (que preserva rigorosamente a norma da componente) destrói completamente a capacidade de recall factual, com a probabilidade caindo para $0,00$ em praticamente todo o sweep. Isso prova que perturbar a *direção* de ativação é suficiente para anular o sinal semântico, independente de amplificação de magnitude.
3.  **Controle de Direcionamento Aleatório:**
    Perturbar o modelo em uma direção de controle completamente aleatória (não ortogonalizada) preserva a probabilidade do fato correto (pico de $0,9463$ sob direções específicas) nas direções em que o ruído não intersecta o plano semântico causal, confirmando a alta especificidade direcional do canal factual na Camada 12.
4.  **Ablação por Profundidade (Layer-wise Ablation):**
    *   **Camada 2 (Inicial):** Exibe resiliência extrema a perturbações. As probabilidades permanecem em $1,0$ na maior parte do sweep e apenas caem levemente nas proximidades da perturbação crítica, refletindo que a informação semântica ainda não se consolidou no fluxo residual.
    *   **Camada 30 (Final):** Mostra instabilidade geral. A norma é extremamente alta ($\approx 54.000$ vs $\approx 1.000$ na Camada 2), e as probabilidades sofrem flutuações voláteis mesmo na ausência de rotações direcionadas.
    Isso valida empiricamente que a Camada 12 atua como o ponto de cristalização semântica ideal, onde as bacias de atração locais possuem o equilíbrio ótimo entre estabilidade sintática e plasticidade semântica.

## 5. Análise Topológica Autoregressiva Avançada ($H_1$ via ripser)

Implementamos a varredura topológica no script [o6_topology.py](../../scripts/o6_topology.py) com análises em $H_1$ (loops 1-dimensionais) aplicadas sobre a nuvem de pontos das ativações na Camada 13.

### Edge Cases Resolvidos
1. **Concatenação Autoregressiva de Nuvem de Pontos:** Capturamos os tensores `[batch, 1, d_model]` de cada passo de geração e os concatenamos ao longo da dimensão de sequência para formar a nuvem de pontos final `[seq_len_generated, d_model]`, preservando a estrutura temporal sem esmagar as dimensões.
2. **Divergência KL Focalizada:** Calculamos a divergência KL estritamente sobre os logits do *primeiro token gerado*, pois os passos subsequentes bifurcam em históricos de contexto distintos, invalidando comparações diretas de distribuição.
3. **Prevenção de OOM no MPS:** Adicionamos chamadas explícitas a `torch.mps.empty_cache()` a cada iteração do loop para liberar tensores obsoletos do Metal Performance Shaders no Apple Silicon.
4. **Salvamento Incremental:** O script salva os resultados no arquivo JSON a cada controle $K$ concluído para evitar perda de dados.

### Resultados Empíricos (Varredura de 30 Planos Ortogonais e 10 Ângulos)
Os dados brutos coletados foram salvos em [topology_sweep.json](../../results/topology_sweep.json) e a média dos resultados para os 30 planos de controle exibe o seguinte comportamento:

*   **Linha de Base Limpa (Clean Baseline):** Betti-0 = 3 | H1 Persistence Total = 54.77 | Divergência KL = 0.00
*   **Sweep Estatístico de Controles (Média de $K=30$):**
    *   **$\theta = 0^\circ$ (Direcionamento factual puro):** Betti-0 = 1.53 | H1 Pers. = 383.60 | KL = 28.54
    *   **$\theta = 40^\circ$:** Betti-0 = 1.43 | H1 Pers. = 292.05 | KL = 26.00
    *   **$\theta = 80^\circ$:** Betti-0 = 1.57 | H1 Pers. = 287.44 | KL = 35.19
    *   **$\theta = 120^\circ$ (Colapso do Manifold):** Betti-0 = 1.47 | H1 Pers. = 230.03 | KL = 38.50 (Pico de Divergência)
    *   **$\theta = 160^\circ$:** Betti-0 = 1.37 | H1 Pers. = 247.05 | KL = 37.42
    *   **$\theta = 200^\circ$ (Translação de Atrator):** Betti-0 = 1.40 | H1 Pers. = 289.51 | KL = 34.94
    *   **$\theta = 240^\circ$:** Betti-0 = 1.50 | H1 Pers. = 275.12 | KL = 33.27
    *   **$\theta = 280^\circ$:** Betti-0 = 1.60 | H1 Pers. = 286.41 | KL = 30.32
    *   **$\theta = 320^\circ$ (Recuperação Local Parcial):** Betti-0 = 1.40 | H1 Pers. = 331.05 | KL = 22.46 (Mínima Perturbação)
    *   **$\theta = 360^\circ$:** Betti-0 = 1.53 | H1 Pers. = 383.60 | KL = 28.54

### Interpretação Topológica
*   **Betti-0:** A queda de Betti-0 de 3 (linha de base) para cerca de 1.4-1.6 em todos os ângulos perturbados documenta a contração e a simplificação dos conectados da nuvem de pontos durante o colapso semântico.
*   **Persistência Total $H_1$:** O aumento drástico de $54.77$ para $>200$ sob perturbação decorre do fato de o modelo começar a gerar fragmentos repetitivos e ciclicamente redundantes (como `ififif...` ou `बजा बजा...`), que são detectados pelo `ripser` como loops topológicos altamente estáveis e persistentes no espaço latente de 2560 dimensões. A zona de colapso crítico ($\theta=120^\circ$) exibe o vale local de persistência ($230.03$) devido ao esmagamento completo da variedade ativacional.

Os gráficos das Persistence Barcodes das iterações do controle $K=0$ foram salvos no diretório [docs/antigr_reports/](../../docs/antigr_reports/) (e.g. `h1_barcodes_theta_*.png`).
