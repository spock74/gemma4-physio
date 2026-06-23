# Tractatus Semiótico Operacional: Segmentos de Fundamentação

## Axioma I: A Identidade Funtorial e Vetorial do Significado

Este axioma dissolve a falsa dicotomia ontológica entre "features esparsas" e "direções latentes" (a crise do _Dictionary Learning_ versus _Representation Engineering_), estabelecendo a natureza puramente geométrica da representação no modelo.

1. **A Variedade de Estado:** O modelo não assimila "conceitos discretos" ou "átomos semânticos". O treinamento estrutura uma Variedade de Estado Topológica (Manifold), denotada como _M_.
2. **Definição Operacional (Isomorfismo de Yoneda):** A identidade de um estado de ativação não possui valor intrínseco. O "significado" é definido estritamente pela totalidade de relações e morfismos (transformações) que esse estado permite dentro do fluxo residual. A semântica é a estrutura das interações no subespaço.
3. **Unificação Vetorial:**
   - Uma _Feature_ é definida restritamente como um vetor base de um subespaço ortogonalizado projetado sobre _M_.
   - Uma _Direção Latente_ é o gradiente direcional de ativação ao longo da variedade _M_.
4. **O Objeto Semiótico:** Um "Conceito Operacional" é a trajetória (geodésica) do fluxo residual quando este é submetido à atração de um vetor base específico dentro do espaço de estados. A interpretação mecanística é o mapeamento fiel dessa topologia geométrica.

## Axioma II: A Dinâmica de Trajetória e o Colapso Causal

Este axioma governa a mecânica de intervenção no modelo e explica o comportamento autocorretivo da rede frente a perturbações não alinhadas.

1. **A Estrutura do Fluxo:** O _residual stream_ é um Grafo Causal Direcionado (DAG) em constante evolução, onde as matrizes de atenção e MLP de cada camada _L_n_ operam como mapeamentos topológicos sucessivos.
2. **A Intervenção Funtorial:** O ato de modificar o raciocínio da rede (como o _Causal Patching_ direcionado a alucinações) é a aplicação do operador de Pearl _do(v⃗)_. Esta operação age como um functor que projeta um vetor de estado forçado sobre o subespaço em _L_n_, suprimindo as arestas causais de entrada originais daquele nó.
3. **Teorema da Sobrescrita (Colapso Causal):** Uma intervenção mecânica só é incorporada se preservar a invariância estrutural. Se a projeção _do(v⃗)_ não comutar com os atratores topológicos programados para a camada subsequente _L\_{n+1}_, o sistema de pesos atuará como um mecanismo corretivo. A rede anulará a modificação, forçando a trajetória de volta para o subespaço original (o atrator patológico ou correto), tratando _v⃗_ como ruído ortogonal.
4. **Métrica de Falsificabilidade:** A estabilidade de uma intervenção na variedade e a ausência de colapso causal devem ser medidas matematicamente através da manutenção da similaridade de cosseno de _v⃗_ no cruzamento com _L\_{n+1}_, acoplada à medição de desvio via Divergência de Kullback-Leibler (KL) na distribuição final estocástica em _W_U_.
