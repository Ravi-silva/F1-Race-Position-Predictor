# -*- coding: utf-8 -*-
"""
Dados REAIS da temporada 2026 de F1, coletados da internet (formula1.com e
fontes jornalísticas) em 01/09/2026 — logo após o GP da Holanda (round 12,
a corrida mais recente disputada até a data), com a Itália (Monza, round 13)
como próxima etapa do calendário.

Este módulo é a "ponte" entre o modelo treinado com dados históricos do
Kaggle (1950-2024, ver notebook.ipynb) e a temporada 2026 atual: ele guarda
o grid de pilotos/equipes REALMENTE em atividade agora, mais o resultado
real do último GP para usar como ponto de partida da previsão.

IMPORTANTE — o "problema do calouro" (cold start):
O modelo nunca viu, durante o treino, 3 pilotos estreantes (Antonelli,
Bortoleto, Lindblad) nem 2 equipes estreantes (Audi, Cadillac) — eles não
existem no dataset 1950-2024. Como o encoder de categorias (LabelEncoder)
só conhece os rótulos vistos no treino, usamos um "proxy" histórico só para
a variável categórica piloto/equipe (ex.: Audi herda o código de "Sauber",
a mesma equipe antes do rebranding). O app deixa isso visível na tela.
Isso pesa pouco no resultado final: grid/quali/forma pesam muito mais que
piloto/equipe codificados (ver feature importance no notebook, seção 12).
"""

# round mais recente já disputado (GP da Holanda, Zandvoort, 21-23/08/2026)
REFERENCE_ROUND = 12
REFERENCE_GP = "GP da Holanda 2026 (Zandvoort)"
REFERENCE_DATE = "23/08/2026"

# próxima corrida do calendário a partir de hoje (01/09/2026)
NEXT_ROUND = 13

# ---------------------------------------------------------------------------
# Calendário 2026 (23 etapas) — round, nome, circuitRef (código do dataset
# histórico), cidade/país, status.
# circuitRef confere com data/circuits.csv (todas as pistas de 2026 já
# tinham corrida no período 1950-2024, então nenhum proxy é necessário aqui).
# ---------------------------------------------------------------------------
CALENDAR_2026 = [
    {"round": 1,  "gp": "GP da Austrália",        "circuit_ref": "albert_park",    "local": "Melbourne"},
    {"round": 2,  "gp": "GP da China",             "circuit_ref": "shanghai",       "local": "Xangai"},
    {"round": 3,  "gp": "GP do Japão",             "circuit_ref": "suzuka",         "local": "Suzuka"},
    {"round": 4,  "gp": "GP de Miami",             "circuit_ref": "miami",          "local": "Miami"},
    {"round": 5,  "gp": "GP do Canadá",            "circuit_ref": "villeneuve",     "local": "Montreal"},
    {"round": 6,  "gp": "GP de Mônaco",            "circuit_ref": "monaco",         "local": "Monte Carlo"},
    {"round": 7,  "gp": "GP da Espanha",           "circuit_ref": "catalunya",      "local": "Barcelona"},
    {"round": 8,  "gp": "GP da Áustria",           "circuit_ref": "red_bull_ring",  "local": "Spielberg"},
    {"round": 9,  "gp": "GP da Grã-Bretanha",      "circuit_ref": "silverstone",    "local": "Silverstone"},
    {"round": 10, "gp": "GP da Bélgica",           "circuit_ref": "spa",            "local": "Spa-Francorchamps"},
    {"round": 11, "gp": "GP da Hungria",           "circuit_ref": "hungaroring",    "local": "Budapeste"},
    {"round": 12, "gp": "GP da Holanda",           "circuit_ref": "zandvoort",      "local": "Zandvoort"},
    {"round": 13, "gp": "GP da Itália",            "circuit_ref": "monza",          "local": "Monza"},
    {"round": 14, "gp": "GP da Espanha (Madrid)",  "circuit_ref": "catalunya",      "local": "Madrid"},
    {"round": 15, "gp": "GP do Azerbaijão",        "circuit_ref": "baku",           "local": "Baku"},
    {"round": 16, "gp": "GP do Bahrein",           "circuit_ref": "bahrain",        "local": "Sakhir"},
    {"round": 17, "gp": "GP de Singapura",         "circuit_ref": "marina_bay",     "local": "Marina Bay"},
    {"round": 18, "gp": "GP dos EUA",              "circuit_ref": "americas",       "local": "Austin"},
    {"round": 19, "gp": "GP do México",            "circuit_ref": "rodriguez",      "local": "Cidade do México"},
    {"round": 20, "gp": "GP do Brasil",            "circuit_ref": "interlagos",     "local": "São Paulo"},
    {"round": 21, "gp": "GP de Las Vegas",         "circuit_ref": "vegas",          "local": "Las Vegas"},
    {"round": 22, "gp": "GP do Catar",             "circuit_ref": "losail",         "local": "Lusail"},
    {"round": 23, "gp": "GP de Abu Dhabi",         "circuit_ref": "yas_marina",     "local": "Yas Island"},
]
# round 14 usa o circuito de Barcelona como referência mais próxima —
# a nova pista de Madrid (Madring) ainda não tinha nome fechado nas fontes
# no momento da coleta e não existe no dataset histórico.

# ---------------------------------------------------------------------------
# Pontos por equipe na temporada 2026 até o round 12 (fonte: classificação
# oficial de construtores). Usado para constructor_form / constructor_season_pts.
# ---------------------------------------------------------------------------
CONSTRUCTOR_POINTS_2026 = {
    "Mercedes":         425,
    "Ferrari":          338,
    "McLaren":          263,
    "Red Bull Racing":  186,
    "Racing Bulls":      66,
    "Alpine":            63,
    "Haas F1 Team":      21,
    "Audi":              16,
    "Williams":          11,
    "Aston Martin":       3,
    "Cadillac":           0,
}

# equipe -> constructorRef do dataset histórico usado para a variável
# categórica codificada. Audi e Cadillac não existem no histórico 1950-2024
# (Audi só passou a se chamar assim em 2026; Cadillac é montadora estreante)
# -> usamos um "proxy": Audi herda "sauber" (mesma equipe antes do
# rebranding); Cadillac (equipe nova, sem herança) usa "haas" como proxy,
# por ser a comparação mais próxima disponível no histórico (outra
# fabricante nova estreando no grid, sem pedigree em F1).
CONSTRUCTOR_REF = {
    "McLaren":          ("mclaren", None),
    "Mercedes":         ("mercedes", None),
    "Red Bull Racing":  ("red_bull", None),
    "Ferrari":          ("ferrari", None),
    "Williams":         ("williams", None),
    "Racing Bulls":     ("rb", None),
    "Aston Martin":     ("aston_martin", None),
    "Haas F1 Team":     ("haas", None),
    "Alpine":           ("alpine", None),
    "Audi":             ("sauber", "Audi (equipe estreante em 2026) → usando o código histórico da Sauber, mesma equipe antes do rebranding"),
    "Cadillac":         ("haas", "Cadillac (montadora estreante em 2026, sem histórico em F1) → usando a Haas como referência aproximada, outra equipe nova sem pedigree"),
}

# ---------------------------------------------------------------------------
# Grid de pilotos 2026 (22 pilotos, 11 equipes) — nomes, equipe atual e
# nacionalidade puxados automaticamente da internet (não são mais escolhidos
# manualmente no app). grid / quali_position / quali_gap / recent_result são
# o resultado REAL do último GP disputado (Holanda, round 12); usados como
# valor inicial das previsões e continuam ajustáveis para simular cenários.
# ---------------------------------------------------------------------------
DRIVERS_2026 = [
    # nome,               driverRef,        proxy,        equipe,             nacionalidade,   grid, quali, gap,   finish, pontos_temporada, experiencia_estimada
    ("Lando Norris",       "norris",          None,        "McLaren",          "British",        1,  1,  0.000,  1,  159, 164),
    ("Oscar Piastri",      "piastri",         None,        "McLaren",          "Australian",     4,  4,  0.142,  6,  104,  82),
    ("George Russell",     "russell",         None,        "Mercedes",         "British",        2,  2,  0.102,  3,  183, 164),
    ("Kimi Antonelli",     "antonelli",       "russell",   "Mercedes",         "Italian",        3,  3,  0.133,  2,  242,  36),
    ("Max Verstappen",     "max_verstappen",  None,        "Red Bull Racing",  "Dutch",          7,  7,  0.455, 22,  112, 245),
    ("Liam Lawson",        "lawson",          None,        "Red Bull Racing",  "New Zealander",  8,  8,  0.570,  7,   49,  47),
    ("Charles Leclerc",    "leclerc",         None,        "Ferrari",          "Monegasque",     6,  6,  0.395,  5,  155, 185),
    ("Lewis Hamilton",     "hamilton",        None,        "Ferrari",          "British",        5,  5,  0.331,  4,  183, 392),
    ("Alexander Albon",    "albon",           None,        "Williams",         "Thai",          16, 16,  1.900, 17,    5, 141),
    ("Carlos Sainz",       "sainz",           None,        "Williams",         "Spanish",       17, 17,  2.050, 16,    6, 244),
    ("Yuki Tsunoda",       "tsunoda",         None,        "Racing Bulls",     "Japanese",      12, 12,  1.300, 11,    0, 126),
    ("Arvid Lindblad",     "lindblad",        "tsunoda",   "Racing Bulls",     "British",       10, 10,  1.022, 12,   23,  12),
    ("Fernando Alonso",    "alonso",          None,        "Aston Martin",     "Spanish",       18, 18,  2.200,  9,    3, 440),
    ("Lance Stroll",       "stroll",          None,        "Aston Martin",     "Canadian",      19, 19,  2.350, 20,    0, 204),
    ("Esteban Ocon",       "ocon",            None,        "Haas F1 Team",     "French",        15, 15,  1.750, 19,    3, 192),
    ("Oliver Bearman",     "bearman",         None,        "Haas F1 Team",     "British",       20, 20,  2.500, 21,   18,  39),
    ("Nico Hulkenberg",    "hulkenberg",      None,        "Audi",             "German",        13, 13,  1.450,  8,    6, 266),
    ("Gabriel Bortoleto",  "bortoleto",       "hulkenberg","Audi",             "Brazilian",      9,  9,  0.916, 13,   10,  36),
    ("Pierre Gasly",       "gasly",           None,        "Alpine",           "French",        11, 11,  1.150, 10,   44, 190),
    ("Franco Colapinto",   "colapinto",       None,        "Alpine",           "Argentinian",   14, 14,  1.600, 14,   19,  45),
    ("Valtteri Bottas",    "bottas",          None,        "Cadillac",         "Finnish",       21, 21,  2.650, 18,    0, 283),
    ("Sergio Perez",       "perez",           None,        "Cadillac",         "Mexican",       22, 22,  2.800, 15,    0, 295),
]
# gap de quali (s) exato para P1-P10 (tempo oficial de Q3); P11-P22 é uma
# estimativa (não fizeram Q3, tempos de Q1/Q2 não abertos pela fonte) —
# sinalizado na interface. driver_experience = corridas na carreira até
# 2024 (contadas no dataset) + estimativa de 2025 e 2026 (temporada cheia
# de 24 corridas para quem já disputou 2025 completo; Sergio Pérez não
# correu em 2025; estreantes contam só 2025-2026 ou só 2026).

DRIVERS_2026 = [
    {
        "name": n, "driver_ref": ref, "driver_proxy": proxy, "team": team,
        "nationality": nat, "grid": g, "quali_position": q, "quali_gap": gap,
        "recent_result": fin, "season_points": pts, "driver_experience": exp,
    }
    for (n, ref, proxy, team, nat, g, q, gap, fin, pts, exp) in DRIVERS_2026
]
