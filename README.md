# F1 Race Position Predictor 🏎️

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)

Modelo de Machine Learning que prevê, **antes da largada de um GP de Fórmula 1**, em que faixa de
resultado cada piloto deve terminar: **pódio**, **zona de pontos** ou **fora dos pontos**.

> **Resultado:** Random Forest com **67,9% de acurácia** no teste, contra 52,5% de um chute trivial,
> treinado com 25 temporadas de dados reais (2000–2024). Inclui um app Streamlit que aplica o modelo
> à temporada **2026** em andamento.

---

## Sumário

- [O problema](#o-problema)
- [Dados](#dados)
- [Metodologia](#metodologia)
- [Resultados](#resultados)
- [App: previsão para 2026](#app-previsão-para-a-temporada-2026)
- [Como rodar](#como-rodar)
- [Estrutura do repositório](#estrutura-do-repositório)
- [Próximos passos](#próximos-passos)

---

## O problema

Acertar a posição exata de chegada (P1, P2, P3…) é quase impossível: safety car, batidas, chuva e
estratégia de pit stop mudam tudo durante a corrida. Por isso o projeto trata a questão como uma
**classificação em 3 classes**, que é o que importa na prática:

| Classe | Significado | Posição final |
|---|---|---|
| 🏆 **Podium** | subiu ao pódio | P1 – P3 |
| 🔵 **Points** | pontuou | P4 – P10 |
| ⚪ **Out of Points** | não pontuou (ou abandonou) | P11 ou pior |

A pergunta que o modelo responde: *"com o que se sabe antes da largada (grid, classificação, forma
recente, força da equipe), esse piloto tem chance de pódio ou de pontuar?"* É a base de uma
ferramenta de preview de GP, de um bolão ou de uma análise pré-corrida.

## Dados

**[Formula 1 World Championship (1950–2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)**,
no Kaggle. É um espelho da antiga API Ergast e, apesar do nome, a versão atual vai até **2024**.

7 tabelas unidas pelas chaves `raceId`, `driverId`, `constructorId`, `circuitId` e `statusId`:

`races` · `results` · `qualifying` · `drivers` · `constructors` · `circuits` · `status`

Os CSVs não são versionados. Veja em [Como rodar](#como-rodar) como baixá-los.

## Metodologia

| Etapa | O que foi feito |
|---|---|
| **Join** | 7 tabelas em torno de `results` (uma linha = um piloto numa corrida) |
| **Escopo** | Era moderna, **2000–2024** |
| **Alvo** | `positionOrder` → Podium / Points / Out of Points |
| **Abandonos (DNF)** | **Mantidos** (~23% das linhas). Abandonar é um resultado ruim real; removê-los enviesaria o alvo |
| **Limpeza** | `grid = 0` (largada do pit lane) → tratado como P20; ordenação cronológica para as features de histórico |
| **Features** | 13 variáveis (abaixo) |
| **Split** | 80/20 estratificado, mantendo a proporção das 3 classes |
| **Modelo** | `RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced")` |
| **Baseline** | Regressão Logística com `StandardScaler` |
| **Persistência** | `pickle` com modelo, encoders, lista de features e métricas |

**Features**

- **Largada:** `grid`, `quali_position`, `quali_time_gap` (diferença para a pole)
- **Piloto:** `recent_form` (média das últimas 3 corridas), `driver_experience`
- **Equipe:** `constructor_form` (pontos nas últimas 5 corridas), `constructor_season_pts` (média na temporada até aqui)
- **Contexto:** `year`, `round`
- **Categóricas (Label Encoding):** `circuit`, `driver`, `constructor`, `nationality`

> **Sem data leakage:** todas as features de histórico usam `.shift()`, então o resultado da
> corrida que está sendo prevista nunca entra no cálculo da sua própria feature.

## Resultados

| Modelo | Acurácia (teste) |
|---|---|
| Chute trivial (sempre "Out of Points") | 0,525 |
| Regressão Logística (baseline) | 0,669 |
| **Random Forest (modelo final)** | **0,679** |

Validação cruzada (5 folds): **0,659 ± 0,035**.

**Desempenho por classe (Random Forest)**

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| Podium | 0,61 | 0,79 | 0,69 |
| Points | 0,55 | 0,64 | 0,60 |
| Out of Points | 0,82 | 0,67 | 0,74 |

**O que os resultados mostram**

- **A largada explica a maior parte do resultado:** `grid` e `quali_position` são as features mais
  importantes, seguidas da força da equipe (`constructor_form`, `constructor_season_pts`) e do
  `quali_time_gap`.
- **O meio do grid é o mais difícil:** a classe *Points* tem o pior F1, porque ali o resultado é
  decidido *durante* a corrida (ultrapassagens, undercut, incidentes), e não pela posição de largada.
- **Random Forest × Logística:** a diferença é pequena (~1 ponto). O sinal principal é quase linear;
  o RF ganha por capturar interações como circuito × grid e piloto × equipe, e tem recall bem maior
  na classe *Points* (0,64 contra 0,50).

## App: previsão para a temporada 2026

O app Streamlit ([app.py](app.py)) usa o modelo treinado com dados históricos para simular corridas
da temporada **2026 em andamento**:

- O **piloto** é escolhido entre os **22 pilotos do grid real de 2026**. **Equipe** e
  **nacionalidade** são preenchidas automaticamente.
- **Grid, classificação e forma recente** já vêm com o resultado **real** do último GP disputado
  (Holanda, round 12, 23/08/2026) e podem ser ajustados nos sliders para testar cenários, como
  *"e se ele largasse em P3?"*.
- O app mostra a classe prevista e a probabilidade de cada uma.

**Cold start dos estreantes.** 3 pilotos (Antonelli, Bortoleto, Lindblad) e 2 equipes (Audi,
Cadillac) não existem nos dados de treino. Para a variável categórica, o app usa uma referência
histórica aproximada (por exemplo, a Audi herda o código da Sauber, que é a mesma equipe antes do
rebranding) e avisa isso na tela. Os dados numéricos desses pilotos continuam reais. Origem dos
dados e aproximações: [f1_2026_data.py](f1_2026_data.py).

## Como rodar

```bash
# 1. ambiente e dependências
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 2. dataset do Kaggle em data/ (requer conta Kaggle configurada para o kagglehub)
python -c "import kagglehub,shutil,glob,os; s=kagglehub.dataset_download('rohanrao/formula-1-world-championship-1950-2020'); os.makedirs('data',exist_ok=True); [shutil.copy(f,'data/') for f in glob.glob(os.path.join(s,'*.csv'))]"

# 3. rodar o notebook inteiro (gera o f1_model.pkl)
jupyter notebook notebook.ipynb

# 4. abrir o app
streamlit run app.py
```

> O arquivo `f1_model.pkl` não é versionado. Rode o notebook uma vez para gerá-lo antes de abrir o app.

## Estrutura do repositório

```
├── notebook.ipynb      # análise completa: join, EDA, features, treino e avaliação (com gráficos)
├── app.py              # app Streamlit: previsão para a temporada 2026
├── f1_2026_data.py     # grid 2026 (pilotos, equipes, nacionalidades) e resultado do último GP
├── apresentacao.html   # apresentação do projeto em slides (abrir no navegador; navegar com as setas)
├── requirements.txt
└── README.md
```

## Stack

**pandas** e **numpy** (manipulação), **scikit-learn** (modelos, encoders, métricas),
**matplotlib** (gráficos), **pickle** (serialização), **Streamlit** e **Plotly** (app).

## Próximos passos

- **Split temporal** (treinar até 2021, testar 2022–2024) para simular o uso real.
- **XGBoost / LightGBM**, que costumam ganhar alguns pontos em dados tabulares.
- **Clima** por corrida, já que chuva embaralha o grid.
- **Histórico do piloto em cada circuito.**
- **Tempos de treino livre** como medida de ritmo puro.
- **Probabilidades calibradas**, para entregar "chance de pódio: 62%" em vez de só a classe.
