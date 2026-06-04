# Rapport de Conduite de Projet Data
## Stock Market Analyst — Plateforme de Prédiction & Analyse de Risque

**Auteure :** [Ksenia DAUTEL]
**Formation :** Data Scientist Machine Learning
**Date :** Mai 2026
**Statut :** Projet de fin d'études — Portfolio Personnel

---

## 1. Contexte et Besoins Métiers

### 1.1 Contexte

L'intelligence artificielle transforme le secteur de l'investissement financier. Des millions d'investisseurs particuliers ont désormais accès à des plateformes de bourse en ligne, mais disposent de peu d'outils pour analyser objectivement le risque et la tendance des actifs qu'ils souhaitent acheter.

L'Autorité des Marchés Financiers (AMF) a publié en avril 2025 une mise en garde sur l'utilisation de l'IA pour investir, soulignant que les outils IA actuels "peuvent se baser sur des données obsolètes, inexactes ou incomplètes" et qu'ils ne permettent pas à l'investisseur de comprendre *pourquoi* une recommandation est faite.

Ce projet répond directement à ces lacunes : construire une plateforme Data Science rigoureuse, transparente et explicable, pensée pour l'investisseur particulier.

### 1.2 Problématique

> **Comment aider un investisseur particulier à mieux évaluer le risque et la tendance d'une action, en s'appuyant sur des modèles ML explicables et des données financières actualisées ?**

### 1.3 Besoins identifiés

| Besoin | Type | Priorité |
|--------|------|----------|
| Accéder à des données financières fiables et à jour | Fonctionnel | Haute |
| Prédire la tendance d'une action (hausse / stable / baisse) | Fonctionnel | Haute |
| Évaluer le niveau de risque/volatilité d'un actif | Fonctionnel | Haute |
| Comprendre pourquoi le modèle fait cette prédiction | Fonctionnel | Haute |
| Consulter les news financières récentes liées à l'action | Fonctionnel | Moyenne |
| Interagir via un chatbot pour poser des questions | Fonctionnel | Moyenne |
| Visualiser les données via un dashboard interactif | Fonctionnel | Moyenne |

### 1.4 Cible utilisateur

**Investisseur particulier** : personne physique souhaitant investir en bourse (actions françaises ou américaines), sans expertise avancée en finance quantitative, qui cherche un second avis objectif basé sur les données avant de prendre une décision.

---

## 2. Audit de l'Existant et Analyse des Solutions Disponibles

### 2.1 Solutions actuelles sur le marché

| Solution | Forces | Limites |
|----------|--------|---------|
| ChatGPT / Mistral (usage général) | Accessible, conversationnel | Pas spécialisé finance, pas de données temps-réel, pas de ML structuré |
| Banques | Réglementés, gérés | Boîte noire, pas d'explicabilité, coût élevé |
| TradingView | Dashboard riche | Pas d'IA prédictive intégrée, pas de RAG, interface complexe |
| Bloomberg Terminal | Très complet | Coût prohibitif (>20 000€/an), réservé aux professionnels |

### 2.2 Adéquation avec les besoins

Aucune solution existante ne combine à la fois :
- des **modèles ML prédictifs** entraînés sur des données structurées
- de l'**explicabilité** (SHAP values)
- de l'**analyse de news** par RAG
- une **interface conversationnelle** accessible au grand public
- le tout dans un **pipeline data complet et auditable**

### 2.3 Opportunité

Ce projet apporte une valeur ajoutée réelle en combinant ces briques dans un outil cohérent, transparent et pédagogique — en accord avec les recommandations de l'AMF sur la vigilance et la compréhension des outils IA.

---

## 3. Solution Technique Retenue

### 3.1 Nom du projet

**FinSight** — *Plateforme de Prédiction & Analyse de Risque pour Investisseurs Particuliers*

### 3.2 Description fonctionnelle

L'utilisateur saisit le ticker d'une action (ex : `AAPL`, `MC.PA`, `TTE.PA`). La plateforme :
1. Récupère et agrège les données financières historiques et macroéconomiques
2. Exécute le pipeline ML pour prédire la tendance et scorer la volatilité
3. Analyse les news récentes via un système RAG
4. Affiche les résultats dans un dashboard interactif avec explications SHAP
5. Permet d'interroger un agent IA en langage naturel pour approfondir l'analyse

### 3.3 Architecture technique

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  yfinance (prix, volumes)    NewsAPI (articles)      FRED (macro)│
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────────────────┐
│                     ML PIPELINE                                  │
│  Feature Engineering  ->  Entraînement  ->  Validation           │
│  • Modèle 1 : Classification tendance (XGBoost)                  │
│    Target : ↑ hausse /  stable /  baisse (à 30 jours)            │
│  • Modèle 2 : Scoring volatilité (régression -> classe risque)   │
│  • SHAP : explicabilité globale (beeswarm) et locale (waterfall) │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                                 │
│  News financières -> Chunking -> Embeddings -> FAISS             │
│  (sentence-transformers + LangChain)                             │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────────────────┐
│                  AI AGENT (LangChain)                            │
│  Tool 1 : Interroger les modèles ML                              │
│  Tool 2 : Rechercher dans la base RAG (news)                     │
│  Tool 3 : Générer un résumé de marché automatique                │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌──────────────────────────────────────────────────────────────────┐
│           DASHBOARD + CHATBOT (Streamlit)                        │
│  Graphiques prix  ·  Prédiction tendance  ·  Score risque        │
│  SHAP plots  ·  Résumé news  ·  Interface chat                   │
└──────────────────────────────────────────────────────────────────┘
```

### 3.4 Stack technique

| Composant | Technologie choisie | Justification |
|-----------|-------------------|---------------|
| Données financières | `yfinance` | Gratuit, fiable, simple |
| Données macro | `fredapi` (FRED) | Taux d'intérêt, inflation |
| News | `NewsAPI` (free tier) | Articles récents en anglais et français |
| Feature engineering | `pandas`, `ta` (technical analysis) | RSI, MACD, Bollinger Bands |
| Modèles ML | `XGBoost`, `scikit-learn` | Performant, compatible SHAP |
| Explicabilité | `SHAP` | Déjà maîtrisé  |
| Tracking expériences | `MLflow` | Bonne pratique MLOps |
| RAG | `LangChain` + `FAISS` + `sentence-transformers` | Maîtrisé  |
| LLM | `Mistral` (API) | Gratuit, multilingue |
| Agent IA | `LangChain Agents` | Maîtrisé  |
| Interface | `Streamlit` | Rapide à déployer, adapté à la data |
| Environnement | `uv` + `pyproject.toml` | Maîtrisé |

---

## 4. Features ML — Détail

### 4.1 Features utilisées pour la prédiction

**Features techniques (calculées sur données historiques) :**
- RSI (Relative Strength Index) sur 14 jours
- MACD et signal MACD
- Bandes de Bollinger (largeur, position du prix)
- Moyenne mobile 20j et 50j (et leur croisement)
- Volume relatif (vs moyenne 20j)
- Rendements passés (1j, 5j, 20j)
- Volatilité historique (20j, 60j)

**Features macroéconomiques :**
- Taux directeur Fed / BCE
- Taux d'inflation (CPI)
- Taux 10 ans US (proxy risque marché)

**Feature NLP :**
- Score de sentiment moyen sur les news des 7 derniers jours (analyse de polarité)

### 4.2 Variables cibles

| Modèle | Variable cible | Type |
|--------|---------------|------|
| Tendance | Rendement J+30 > +3% -> hausse / < -3% -> baisse / sinon stable | Classification multiclasse (3 classes) |
| Volatilité | Volatilité réalisée J+30 discrétisée en 3 niveaux | Classification (faible / moyen / élevé) |

---

## 5. Plan de Réalisation

### 5.1 Découpage en sprints

| Sprint | Durée | Tâches | Livrable |
|--------|-------|--------|----------|
| **Sprint 1** | Semaine 1 | Setup environnement, collecte données, EDA | Notebook EDA + données nettoyées |
| **Sprint 2** | Semaine 2 | Feature engineering, premier modèle (baseline) | Notebook ML v1 + métriques baseline |
| **Sprint 3** | Semaine 3 | Optimisation modèles, SHAP, validation croisée | Notebook ML final + SHAP plots |
| **Sprint 4** | Semaine 4 | Pipeline RAG (news -> embeddings -> FAISS) | RAG fonctionnel + évaluation |
| **Sprint 5** | Semaine 5 | Agent IA (LangChain), intégration des outils | Agent fonctionnel |
| **Sprint 6** | Semaine 6 | Dashboard Streamlit + chatbot + tests bout-en-bout | Application complète |
| **Sprint 7** | Semaine 7 | Documentation, rapport, préparation soutenance | Portfolio + rapport final |


---

## 6. Métriques et Validation

### 6.1 Métriques ML

**Modèle de classification tendance :**
- F1-score pondéré (adapté au déséquilibre des classes)
- Matrice de confusion (3x3)
- Courbe précision-rappel par classe
- AUC-ROC par classe (one-vs-rest)
- Comparaison avec un modèle Dummy (baseline)

**Démarche de validation :**
- Walk-forward cross-validation (adapté aux séries temporelles — évite le data leakage)
- Séparation temporelle : entraînement sur données 2015-2022, test sur 2023-2024

### 6.2 Évaluation RAG

- Cohérence des chunks récupérés vs question posée
- Test de questions connues avec réponses attendues
- Vérification manuelle sur un échantillon de 20 questions

### 6.3 Critères de succès du projet

| Critère | Seuil acceptable |
|---------|-----------------|
| F1-score tendance (classe dominante) | > 0.55 |
| F1-score volatilité | > 0.60 |
| Agent IA répond correctement | > 80% des questions de test |
| Dashboard fonctionnel pour 5 tickers différents | 100% |
| SHAP plots générés et interprétables | 100% |


---

## 7. Aspects Éthiques et Réglementaires

Conformément aux recommandations de l'AMF :
- Le système affiche un **disclaimer explicite** : "Cet outil est à but éducatif et ne constitue pas un conseil en investissement."
- Les prédictions sont systématiquement accompagnées de leur **niveau de confiance** et des **SHAP values** permettant à l'utilisateur de comprendre le raisonnement du modèle.
- Aucune donnée personnelle de l'utilisateur n'est collectée.
- Le système ne recommande jamais d'acheter ou vendre un titre de façon directe.

---

## 8. Compétences Démontrées

| Compétence du référentiel | Comment elle est démontrée dans ce projet |
|--------------------------|------------------------------------------|
| Collecter les besoins métiers | Analyse du contexte AMF, définition de la cible utilisateur, tableau des besoins |
| Auditer la solution data | Comparaison des solutions existantes, analyse des sources de données |
| Identifier une solution technique | Architecture complète, choix et justification de la stack |
| Appui stratégique et méthodologique | Justification des choix de validation (walk-forward CV), gestion du data leakage |
| Contrôler et analyser le projet | Plan de sprints, métriques de succès, tableau des risques |

---

## 9. Références

- AMF (2025). *Utiliser l'intelligence artificielle pour investir : à quoi faut-il faire attention ?* https://www.amf-france.org
- Yahoo Finance API (`yfinance`) — https://github.com/ranaroussi/yfinance
- FRED Economic Data — https://fred.stlouisfed.org
- LangChain Documentation — https://docs.langchain.com
- SHAP Documentation — https://shap.readthedocs.io
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.

---

*Document rédigé dans le cadre du projet de fin d'études — Formation Data Scientist Machine Learning*
