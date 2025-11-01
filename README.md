DataAnalyser.py

App desktop em Python + Tkinter para análise rápida de datasets.
Você carrega um CSV local, uma pasta (pega o primeiro .csv) ou um dataset do Kaggle Hub (ex.: jockeroika/life-style-data) e o app entrega:

Matriz de correlação (Pearson) com zoom e scroll

Lista das maiores correlações (com limiar |r| configurável)

Gráfico X·Y com reta de regressão e r; exporta PNG/SVG/PDF

Estatísticas descritivas (média, desvio, quartis, skew etc.)

Histograma com linhas de média e mediana

Calculadora de probabilidade para Normal (P(X<x), P(X>x), P(a<X<b))

Ajuda embutida (glossário)

Bloco de notas (salva em .txt)

Tudo num arquivo só: DataAnalyser.py.

Requisitos

Python 3.10+

Bibliotecas:

pandas, numpy, matplotlib, tkinter (vem com o Python), kagglehub

Opcional: scipy (se não tiver, o app usa um fallback pra norm.cdf)

Instalação rápida:

pip install pandas numpy matplotlib kagglehub scipy


Se não quiser scipy, pode pular — o código já cai no fallback.

Como rodar
python DataAnalyser.py


Ao abrir, a janela pede a fonte do dataset. Você pode:

Selecionar um arquivo .csv

Selecionar uma pasta (o app usa o primeiro .csv encontrado)

Informar um ID do Kaggle Hub, ex.:

jockeroika/life-style-data


Ao usar Kaggle Hub, o app baixa o dataset pro cache local automaticamente.

Uso rápido (o que cada aba faz)
1) Matriz de correlação

Mostra a correlação de todas as colunas numéricas.

Controle de zoom (− / + / Reset) e scroll quando ficar grande.

2) Maiores correlações

Informe o limiar de |r| (ex.: 0.50) e gere a lista ordenada.

Duplo clique numa linha → abre direto o gráfico X·Y da dupla.

3) Correlação X · Y

Escolha X e Y (apenas numéricas).

Mostra dispersão, reta de regressão e r.

Botão “Salvar gráfico” exporta PNG/SVG/PDF (DPI alto).

4) Estatística e Probabilidade

Selecione a variável. O app mostra estatísticas descritivas.

Para numéricas, exibe histograma + calculadora Normal:

P(X < a), P(X > b) ou P(a < X < b).

Ajuda e Notas

? abre glossário de termos.

✎ abre bloco de notas (salva em .txt).

Dicas e atalhos

Sem colunas numéricas? As abas avisam em vez de quebrar.

Lista → X·Y: duplo clique numa correlação para plotar.

A janela já vem redimensionada pra caber gráficos sem sobras.

O export do gráfico usa DPI=300 por padrão.

Estrutura do projeto
.
├── DataAnalyser.py    # app completo (GUI + análises)
└── README.md          # este arquivo

Problemas comuns

Tkinter não abre / backend do Matplotlib

O app já força TkAgg. Se der erro, verifique instalação do Tk:

Windows: normalmente já vem OK com Python oficial.

Linux: instale pacotes tipo python3-tk/tk-dev.

macOS ARM: use Python do python.org ou brew com Tk atualizado.

SciPy faltando

Sem crise: o app usa fallback pra norm.cdf.

Se quiser SciPy: pip install scipy.

Kaggle Hub

Precisa de internet e espaço em disco. Se um dataset não baixar, confirme o ID.

Exemplo de dataset (Kaggle Hub)

Você pode iniciar testando com:

jockeroika/life-style-data

Contribuindo

Abra uma issue com bug/ideia/sugestão.

Faça PR limpo, com descrição do problema/solução e testes básicos.

Licença

Defina uma licença (ex.: MIT) e adicione o arquivo LICENSE.
Se não declarar, o padrão é todos os direitos reservados.

Roadmap (curto)

 Exportar CSV com a lista de maiores correlações

 Seleção múltipla de pares X·Y

 Opção de filtro de colunas por padrão/regex

 Suporte a outros formatos (Parquet/Excel)
