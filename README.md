# Sistemas Nebulosos - Trabalhos Práticos
### Daniel Piçarro e Jesimon Barreto

### Trabalho Prático 1 - Implementação do Fuzzy C-Means (FCM)

1. Como referência para o desenvolvimento deste trabalho, usar as notas de aula(slides) disponíveis no Moodle, bem como a parte introdutória, Seções 15.1, 15.2 e
15.3 do Capítulo 15 - Data Clustering Algorithms, do livro texto Neuro-Fuzzy and Soft Computing, cujo link também encontra-se no Moodle.

2. Fuzzy C-Means: Implemente o algoritmo de agrupamento Fuzzy C-Means (FCM). Caso seja conveniente, modifique o código do algoritmo K-Means fornecido no Moodle;

3. Validação do FCM: Valide o algoritmo FCM com a base de dados “FCMdataset.csv”. Para a validaçãoo, plote os centros dos clusters encontrados pelo algoritmo
FCM sobre a base de dados fornecida.

4. Segmentação de Imagens por Região: Use o algoritmo FCM para fazer segmentação semântica das imagens RGB fornecidas no diretório ImagensTeste do Moodle. Para cada imagem, escolha o número de clusters de forma empírica, com base na observação do número de regiões distintas (em termos de tonalidades de cor) que a imagem possui. Após obter a matriz de partição U, resultado da aplicação do FCM em cada imagem, use esta matriz para colorir cada região (cluster) com a tonalidade do pixel que corresponde ao centro da região. Os pixels 1 que apresentarem maior grau de compatibilidade (pertinência) a uma dada região devem ser coloridos com a tonalidade do pixel central daquela região.

5. Apresentação dos Resultados: Faça um relatório descrevendo suas decisões de implementação, testes realizados e resultados obtidos.
