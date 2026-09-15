
<p align="center">
  <img src="assets/logo_digitai.png" alt="Logo DigitAI" width="180">
</p># DigitAI — Classificador de Dígitos Manuscritos (MNIST)

## Sobre o projeto

O **DigitAI** é um pipeline de Machine Learning para classificação de dígitos manuscritos do dataset MNIST. O projeto percorre as etapas de análise exploratória, pré-processamento, treinamento e avaliação de modelos, além de testes de robustez e inferência com imagens manuscritas produzidas fora do dataset.

Além do cenário padrão de classificação, o projeto investiga:

- o comportamento do modelo diante de classes que não apareceram no treinamento;
- a capacidade de inferir imagens manuscritas produzidas pelo próprio estudante.

## Objetivos

- Explorar a estrutura das imagens do MNIST.
- Preparar os dados com divisão estratificada e normalização dos pixels.
- Comparar os modelos KNN, Random Forest e MLP.
- Avaliar os modelos utilizando Accuracy, Precision, Recall, F1-Score e matrizes de confusão.
- Testar o comportamento do modelo diante de classes não vistas durante o treinamento.
- Classificar imagens próprias dos dígitos 4, 7 e 9.

## Tecnologias utilizadas

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Jupyter Notebook

## Estrutura do repositório

```text
digit-ai/
│
├── assets/
│   ├── logo_digitai.png
│   ├── digito_4.png
│   ├── digito_7.png
│   └── digito_9.png
│
├── digitai_mnist.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Etapas implementadas

### Fase 1 — Carregamento e análise exploratória

O dataset MNIST é carregado e são verificadas as dimensões das matrizes, a distribuição das classes e exemplos visuais dos dígitos.

Cada imagem possui dimensão 28 × 28 pixels e é representada como um vetor com 784 características.

Também são apresentadas visualizações para observar a variação das formas de escrita dos dígitos.

### Fase 2 — Pré-processamento

A base é dividida de forma estratificada em:

- **Treino:** 49.000 imagens.
- **Validação:** 7.000 imagens.
- **Teste:** 14.000 imagens.

Os pixels, originalmente na escala de 0 a 255, são normalizados para o intervalo entre 0 e 1.

A divisão estratificada preserva a distribuição das classes nos diferentes conjuntos.

### Fase 3 — Modelos

Foram treinados três classificadores:

- **KNN:** `n_neighbors=3` e `weights="distance"`.
- **Random Forest:** `n_estimators=100`, `max_depth=20` e `random_state=42`.
- **MLP:** uma camada oculta com 128 neurônios, função de ativação ReLU, `max_iter=30` e `random_state=42`.

Os modelos foram treinados utilizando os dados normalizados.

### Fase 4 — Avaliação comparativa

Os modelos foram avaliados no conjunto de teste utilizando Accuracy, Precision ponderada, Recall ponderado e F1-Score ponderado.

Os resultados foram:

| Modelo | Accuracy | Precision ponderada | Recall ponderado | F1-Score ponderado |
|---|---:|---:|---:|---:|
| KNN | 0,972143 | 0,972371 | 0,972143 | 0,972110 |
| Random Forest | 0,964786 | 0,964787 | 0,964786 | 0,964760 |
| **MLP** | **0,979929** | **0,979962** | **0,979929** | **0,979927** |

A **MLP apresentou o melhor desempenho geral** entre os três modelos avaliados.

Em relação ao tempo de treinamento:

| Modelo | Tempo aproximado |
|---|---:|
| KNN | 0,10 s |
| Random Forest | 42,86 s |
| MLP | 38,70 s |

O KNN apresentou o menor tempo de treinamento. A Random Forest levou aproximadamente 42,86 segundos, enquanto a MLP levou aproximadamente 38,70 segundos.

As matrizes de confusão mostram que os erros se concentram principalmente entre alguns dígitos visualmente semelhantes. No KNN, destacam-se confusões como 4 e 9, 8 e 5, e 2 e 7. No Random Forest, aparecem principalmente confusões entre 4 e 9, 3 e 2, e 8 e 9. A MLP apresentou, de modo geral, menor quantidade de erros.

### Fase 5 — Testes de robustez

#### 5.1 — Classes ocultadas

Os dígitos **4 e 7** foram removidos completamente do conjunto de treinamento.

A MLP restrita foi treinada somente com as classes:

`0, 1, 2, 3, 5, 6, 8 e 9`.

Os conjuntos de validação e teste originais foram mantidos.

#### 5.2 — Teste com classes não vistas

A MLP restrita foi testada exclusivamente com imagens reais dos dígitos 4 e 7, que não estavam presentes no treinamento.

O modelo apresentou **0,0% de Accuracy** para essas classes, pois nenhuma das previsões correspondeu às classes reais.

As imagens do dígito 4 foram classificadas principalmente como 9, enquanto as imagens do dígito 7 também foram classificadas principalmente como 9.

O resultado evidencia uma limitação importante do classificador: quando recebe uma imagem pertencente a uma classe que não foi apresentada durante o treinamento, o modelo não possui uma categoria específica para representar que aquela classe é desconhecida e é obrigado a atribuí-la a uma das classes conhecidas.

Também foi observado o fenômeno de **falsa certeza (overconfidence)**, no qual uma previsão incorreta pode receber uma probabilidade elevada.

#### 5.3 — Imagens próprias

Foram utilizadas imagens próprias dos dígitos 4, 7 e 9, produzidas no Paint com fundo preto e traços claros.

As imagens passaram por um pipeline de pré-processamento para adequação ao formato esperado pelo MNIST, incluindo:

- conversão para escala de cinza;
- remoção de pixels muito escuros;
- identificação do bounding box do dígito;
- centralização do desenho;
- redimensionamento para 28 × 28 pixels;
- normalização dos pixels para o intervalo `[0.0, 1.0]`.

Como as imagens já possuíam fundo preto e traços claros, **não foi necessária a inversão das cores**.

As previsões obtidas pela MLP original foram:

| Dígito real | Previsão |
|---:|---:|
| 4 | **4** |
| 7 | 3 |
| 9 | 3 |

Além das previsões, o notebook apresenta as imagens processadas e os gráficos com as probabilidades de saída para as classes.

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/acaciarosar/digit-ai.git
cd digit-ai
```

### 2. Criar o ambiente virtual

No Windows PowerShell:

```powershell
python -m venv venv
```

### 3. Ativar o ambiente virtual

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Instalar as dependências

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Abrir o notebook

```powershell
jupyter notebook
```

Depois, abra `digitai_mnist.ipynb` e execute as células em ordem.

O notebook realiza o carregamento do dataset MNIST durante a execução. O tempo de treinamento pode variar de acordo com o computador utilizado.

## Observações de reprodutibilidade

- O notebook utiliza `random_state=42` nas divisões e nos modelos em que esse parâmetro é aplicável.
- A MLP pode emitir um aviso de convergência porque o experimento utiliza `max_iter=30`.
- As imagens próprias utilizadas no Desafio C devem permanecer na pasta `assets/`.
- Os caminhos utilizados no notebook são relativos ao diretório do projeto.

## Melhorias futuras

- Aumentar a quantidade de imagens próprias para cada classe.
- Testar uma Rede Neural Convolucional (CNN).
- Ajustar os hiperparâmetros dos modelos.
- Criar um script Python separado para inferência de novas imagens.
- Adicionar mecanismos de detecção de baixa confiança e de entradas fora da distribuição conhecida.
- Melhorar o tratamento de centralização e remoção de margens das imagens produzidas fora do dataset.

## Vídeo de apresentação

Link para o vídeo no Google Drive:

[Link do vídeo no Google Drive](https://drive.google.com/drive/folders/1oPZfFas8ZKg8X6KjENbVhinCdiILVrGU?usp=drive_link)

## Autoria

Projeto desenvolvido por **Acácia Rosar** como atividade avaliativa do Módulo 2 — Desenvolvimento de IA para Análise Preditiva.