# F1 Race Position Predictor 🏎️

Modelo de Machine Learning que prevê, **antes de uma corrida de Fórmula 1**, em que faixa de
resultado cada piloto deve terminar.

---

## Visão geral

Em vez de tentar adivinhar a posição exata de chegada (P1, P2, P3, …) — quase impossível por causa
de safety cars, batidas, chuva e estratégia — o projeto trata o problema como uma **classificação
em 3 classes**:

| Classe | Significado | Posição final |
|---|---|---|
| **Podium** | subiu ao pódio | P1 – P3 |
| **Points** | pontuou | P4 – P10 |
| **Out of Points** | fora dos pontos (ou abandonou) | P11 ou pior |

## O problema que resolve

Dá para responder, com dados históricos e informação disponível **antes da largada** (grid,
classificação, forma recente, força da equipe), a pergunta prática: *"esse piloto tem chance de
pódio / de pontuar nesta corrida?"*. É a base de qualquer ferramenta de preview de GP, bolão,
ou análise pré-corrida.

## Fonte dos dados

**[Formula 1 World Championship (1950-2020)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020)** — Kaggle
(espelho da antiga API Ergast, que foi descontinuada). Apesar do nome, a versão atual do dataset
vai até **2024**.

Tabelas usadas (coloque os `.csv` numa pasta `data/`):

`races.csv`, `results.csv`, `qualifying.csv`, `drivers.csv`, `constructors.csv`,
`circuits.csv`, `status.csv`

Joins pelas chaves `raceId`, `driverId`, `constructorId`, `circuitId`, `statusId`.

## Metodologia (resumo)

1. **Join** das 7 tabelas em torno de `results.csv` (uma linha = um piloto numa corrida).
2. **Alvo**: `positionOrder` → Podium / Points / Out of Points.
3. **Escopo**: era moderna, **2000–2024**.
4. **DNFs (abandonos)**: **mantidos** (≈23% das linhas). Um abandono é um resultado ruim real
   ("Out of Points") e removê-lo enviesaria o alvo. Uma flag `finished` acompanha o efeito.
   Ver discussão no notebook.
5. **Limpeza**: `grid = 0` (pit lane / sem tempo) → tratado como P20; datas convertidas e dataset
   ordenado cronologicamente para as features de histórico.
6. **Features** (13): `grid`, `quali_position`, `quali_time_gap`, `recent_form` (média móvel das
   últimas 3 corridas do piloto), `driver_experience`, `constructor_form` (pontos da equipe nas
   últimas 5 corridas), `constructor_season_pts` (pontos médios da equipe na temporada até aqui),
   `year`, `round`, e `circuit` / `driver` / `constructor` / `nationality` com **Label Encoding**.
   Todas as features de histórico usam `.shift()` para não vazar a corrida atual (*no data leakage*).
7. **Split** estratificado 80/20 (mantém a proporção das 3 classes).
8. **Modelo**: `RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced")`.
9. **Baseline**: Regressão Logística (com `StandardScaler`).
10. **Persistência**: `pickle` (modelo + encoders + lista de features + métricas).

## Como rodar

```bash
# 1. criar ambiente e instalar dependências
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# 2. baixar o dataset do Kaggle e colocar os CSVs em data/
#    (ou, com conta Kaggle configurada:)
python -c "import kagglehub,shutil,glob,os; s=kagglehub.dataset_download('rohanrao/formula-1-world-championship-1950-2020'); os.makedirs('data',exist_ok=True); [shutil.copy(f,'data/') for f in glob.glob(os.path.join(s,'*.csv'))]"

# 3. abrir o notebook
jupyter notebook notebook.ipynb      # Runtime → Run all

# 4. (bônus) app interativo — previsão para a temporada 2026 ATUAL
streamlit run app.py
```

Estrutura:

```
f1-race-position-predictor/
├── notebook.ipynb        # análise completa, do join à avaliação (executado, com gráficos)
├── README.md             # este arquivo
├── entrevista.md         # guia para apresentar o projeto numa entrevista
├── apresentacao.html     # apresentação em 9 slides (abrir no navegador; setas ↑ ↓ / espaço)
├── app.py                # app Streamlit — previsão para a temporada 2026
├── f1_2026_data.py        # grid de pilotos/equipes/nacionalidades 2026 + resultado real do último GP (coletado da internet)
├── requirements.txt
├── f1_model.pkl          # gerado pelo notebook (modelo + encoders + metadados)
└── data/                 # CSVs do Kaggle (não versionados)
```

### O app (bônus): previsão para a temporada 2026 em andamento

O modelo é treinado só com dados históricos (1950-2024, acima), mas o app usa esse modelo
treinado para prever a temporada **2026 atual**:

- O **piloto** é escolhido numa lista fechada com os **22 pilotos realmente no grid 2026**
  (não é mais um piloto qualquer do histórico de 75 anos).
- **Equipe** e **nacionalidade** não são mais escolhidas à mão: são preenchidas automaticamente
  a partir do piloto selecionado.
- **Grid, classificação e forma recente** vêm pré-preenchidos com o resultado **real** do
  último GP disputado (Holanda, round 12, 23/08/2026 — coletado da internet), e continuam
  ajustáveis nos sliders para simular outros cenários (ex.: "e se ele largasse em P3?").
- **Cold start dos estreantes de 2026**: 3 pilotos (Antonelli, Bortoleto, Lindblad) e 2 equipes
  (Audi, Cadillac) não existem no dataset de treino 1950-2024. O app usa um piloto/equipe
  histórico como referência aproximada só para essa variável categórica (ex.: Audi herda o
  código da Sauber, mesma equipe antes do rebranding) — e mostra isso na tela. Os dados
  numéricos desses pilotos (grid, quali, forma) continuam 100% reais.

Detalhes de origem dos dados de 2026 e das aproximações usadas: ver `f1_2026_data.py`.

## Principais resultados

| Modelo | Acurácia (teste) |
|---|---|
| Chute trivial (sempre "Out of Points") | ~0.52 |
| Regressão Logística (baseline) | **~0.67** |
| **Random Forest (modelo final)** | **~0.68** |

Validação cruzada (5 folds): 0.66 ± 0.03.

- **Features mais importantes**: `grid` e `quali_position` (a largada explica a maior parte do
  resultado), seguidas da força da equipe (`constructor_form`, `constructor_season_pts`) e do
  `quali_time_gap`.
- **Onde erra mais**: a classe **"Points"** (meio do grid), onde o resultado é decidido *durante*
  a corrida — ultrapassagens, undercut, incidentes — e não pela posição de largada.
- **Random Forest × Logística**: diferença pequena (~1 ponto). O sinal dominante do problema é
  quase linear; o RF ganha por capturar interações (circuito × grid, piloto × equipe).

## Stack

- **pandas**, **numpy** — manipulação e cálculo
- **scikit-learn** — `RandomForestClassifier`, `LogisticRegression`, `LabelEncoder`,
  `train_test_split`, métricas
- **matplotlib** — gráficos (EDA e resultados)
- **pickle** — serialização do modelo
- **streamlit** + **plotly** — app interativo (bônus)

## Melhorias futuras

- **Dados de clima** por corrida (chuva embaralha completamente o grid).
- **Tempos de volta** de classificação e de treino livre como proxy de ritmo puro.
- Testar **XGBoost / LightGBM** — gradient boosting costuma ganhar alguns pontos neste tipo de tabela.
- Feature de **histórico do piloto naquele circuito específico**.
- **Split temporal** (treinar até 2021, testar 2022-2024) para simular uso real em produção.
- **Calibrar probabilidades** e entregar "chance de pódio: 62%" em vez de apenas a classe.
