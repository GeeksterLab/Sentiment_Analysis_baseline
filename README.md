# Amazon Reviews Sentiment Analysis

![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)
[![Click Here](https://img.shields.io/badge/Click%20Here-blue?style=for-the-badge)](https://amazonreviews-sentimentanalysis.streamlit.app/)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge\&logo=python\&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-005571.svg?style=for-the-badge\&logo=fastapi)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge\&logo=tensorflow\&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge\&logo=keras\&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=for-the-badge\&logo=pandas\&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154f3c?style=for-the-badge)
![Gensim](https://img.shields.io/badge/Gensim-00A98F?style=for-the-badge)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge\&logo=openai\&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge\&logo=jupyter\&logoColor=white)

Projet NLP de **classification de sentiments sur des avis Amazon**.

L'objectif est de prédire automatiquement le sentiment associé à un texte selon trois classes :

* `0` → **Negative**
* `1` → **Neutral**
* `2` → **Positive**

Le projet compare plusieurs approches de classification :

* un CNN utilisant une représentation basée sur la **tokenization Keras** ;
* un CNN utilisant des embeddings **Word2Vec** ;
* un **LLM** utilisé comme modèle de comparaison ;
<!-- * des embeddings **Sentence Transformers** pour explorer la similarité et la recherche sémantique. -->

Une API **FastAPI** permet ensuite d'exposer le modèle sélectionné pour effectuer des prédictions sur de nouveaux textes.

Une interface **Streamlit** est prévue afin de permettre à un utilisateur d'écrire directement une phrase ou un avis et d'obtenir son sentiment prédit.

---

## Objectifs

Le projet cherche à explorer plusieurs problématiques classiques du NLP :

* nettoyage et préparation de données textuelles ;
* classification multiclasse ;
* gestion du déséquilibre entre classes ;
* tokenization et padding ;
* apprentissage d'embeddings ;
* comparaison de différentes représentations du texte ;
* comparaison entre un modèle Deep Learning local et un LLM ;
* exposition d'un modèle NLP via une API ;
<!-- * recherche sémantique à partir d'embeddings. -->

---

## Dataset

Le projet utilise un dataset d'avis Amazon contenant notamment :

* le texte de l'avis ;
* son score ;
* différentes informations associées à l'avis.

Le score original compris entre `1` et `5` est transformé en trois classes de sentiment :

| Score   | Classe | Sentiment |
| ------- | ------ | --------- |
| `1 - 2` | `0`    | Negative  |
| `3`     | `1`    | Neutral   |
| `4 - 5` | `2`    | Positive  |

Le projet analyse également la distribution des scores et des sentiments afin d'identifier le déséquilibre entre les différentes classes.

---

## Preprocessing

Les textes sont nettoyés avant leur utilisation par les modèles.

Le preprocessing comprend notamment :

* conversion en minuscules ;
* suppression des balises HTML ;
* suppression des URLs ;
* suppression de la ponctuation ;
* suppression des espaces multiples ;
* tokenization avec NLTK ;
* suppression des stopwords.

Les textes nettoyés sont ensuite utilisés pour créer les représentations nécessaires aux différents modèles.

---

## Tokenization

Pour le modèle principal, les textes sont transformés en séquences numériques grâce au tokenizer Keras.

Configuration utilisée :

```text
VOCAB_SIZE = 20000
MAX_LEN = 200
OOV_TOKEN = "<OOV>"
EMBEDDING_DIM = 100
```

Le tokenizer :

1. apprend le vocabulaire sur les données d'entraînement ;
2. transforme les textes en séquences d'identifiants ;
3. gère les mots inconnus avec un token `<OOV>` ;
4. applique un padding pour obtenir des séquences de longueur fixe.

Le tokenizer entraîné est sauvegardé dans :

```text
models/tokenizer_a.pkl
```

---

## Modèles

### Baseline A — CNN + Tokenization

Le premier modèle utilise une couche d'embedding apprise directement pendant l'entraînement.

Architecture :

```text
Input
  ↓
Embedding
  ↓
Conv1D
  ↓
GlobalMaxPooling1D
  ↓
Dense
  ↓
Softmax
  ↓
Negative / Neutral / Positive
```

Le modèle utilise notamment :

* `Embedding`
* `Conv1D`
* `GlobalMaxPool1D`
* `Dense`
* activation finale `softmax`

La fonction de perte utilisée est :

```text
sparse_categorical_crossentropy
```

Le meilleur modèle est sauvegardé dans :

```text
models/cnn_tokenization.keras
```

---

### Baseline B — CNN + Word2Vec

Le second modèle conserve une architecture CNN similaire mais initialise sa couche d'embedding à partir de vecteurs **Word2Vec** entraînés avec Gensim.

Word2Vec permet d'apprendre une représentation vectorielle des mots à partir de leur contexte.

Configuration principale :

```text
vector_size = 100
window = 5
sg = 1
epochs = 10
```

`sg=1` correspond à l'utilisation de l'architecture **Skip-Gram**.

Le modèle final suit ensuite l'architecture :

```text
Word2Vec Embeddings
        ↓
Embedding Layer
        ↓
Conv1D
        ↓
GlobalMaxPooling1D
        ↓
Dense
        ↓
Softmax
```

Le modèle est sauvegardé dans :

```text
models/cnn_word2vec.keras
```

---

## Gestion du déséquilibre des classes

Les sentiments ne sont pas nécessairement répartis uniformément dans le dataset original.

Le projet utilise donc `compute_class_weight` de scikit-learn afin de calculer des poids adaptés à chaque classe.

Ces poids sont transmis au modèle pendant l'entraînement afin d'éviter qu'il privilégie excessivement la classe majoritaire.

Une expérimentation utilise également un échantillon stratifié et équilibré d'environ **50 000 reviews**.

---

## Entraînement

Les modèles CNN utilisent notamment :

```text
Optimizer: RMSprop
Loss: sparse_categorical_crossentropy
Metric: accuracy
```

Plusieurs callbacks sont utilisés :

* `EarlyStopping`
* `ModelCheckpoint`
* `ReduceLROnPlateau`

Ils permettent respectivement :

* d'arrêter l'entraînement lorsque la validation ne progresse plus ;
* de conserver la meilleure version du modèle ;
* de diminuer automatiquement le learning rate lorsque la loss stagne.

---

## Comparaison des modèles

Le notebook compare :

```text
CNN Tokenization
       VS
CNN Word2Vec
       VS
LLM
```

Les deux CNN sont comparés à partir notamment de :

* training accuracy ;
* validation accuracy ;
* training loss ;
* validation loss ;
* precision ;
* recall ;
* F1-score.

Les résultats expérimentaux du notebook montrent que la **Baseline A — Tokenization** obtient de meilleures performances de validation que la **Baseline B — Word2Vec**, notamment avec une validation accuracy plus élevée et une validation loss plus faible.

---

## Comparaison avec un LLM

Le projet utilise également un LLM comme système de classification de référence.

Le prompt demande au modèle de retourner uniquement :

```text
0 = negative
1 = neutral
2 = positive
```

Exemple :

```text
Classify the sentiment of the following Amazon review.

Return only one integer:
0 = negative
1 = neutral
2 = positive

Do not return any explanation.
```

Le projet mesure également le nombre de tokens nécessaires afin d'estimer le coût d'une classification massive avec un LLM.

L'objectif est de comparer deux stratégies différentes :

| CNN local                 | LLM                                           |
| ------------------------- | --------------------------------------------- |
| Inférence locale          | API externe                                   |
| Pas de coût par requête   | Coût par token                                |
| Faible latence            | Latence réseau + génération                   |
| Nécessite un entraînement | Peut fonctionner sans entraînement spécifique |
| Modèle spécialisé         | Modèle généraliste                            |

Cette comparaison permet de réfléchir au compromis entre **performance, coût, latence et simplicité de déploiement**.

---

## Recherche sémantique

Le notebook explore également une seconde composante NLP basée sur :

```text
SentenceTransformer("all-MiniLM-L6-v2")
```

Chaque review peut être transformée en un embedding dense de **384 dimensions**.

Exemple :

```text
Review
  ↓
Sentence Transformer
  ↓
384-dimensional embedding
```

Ces représentations peuvent ensuite être utilisées pour développer une fonctionnalité de **recherche sémantique**, par exemple afin de retrouver les avis les plus proches d'une phrase donnée.

Cette partie est actuellement expérimentale et pourra être intégrée à une version future de l'application.

---

## API FastAPI

L'API charge le modèle CNN Tokenization au démarrage :

```text
models/cnn_tokenization.keras
```

Le modèle reste ensuite disponible dans l'état de l'application afin d'éviter de le recharger pour chaque requête.

---

## Routes API

| Méthode | Route            | Description                                       |
| ------- | ---------------- | ------------------------------------------------- |
| `GET`   | `/health`        | Vérifie le statut de l'API                        |
| `POST`  | `/predict`       | Prédit le sentiment d'un texte                    |
| `POST`  | `/upload-csv`    | Réception d'un fichier CSV                        |
| `POST`  | `/predict-batch` | Prediction de plusieurs textes — en développement |

---

## Prediction unitaire

La route principale est :

```text
POST /predict
```

Elle reçoit :

```json
{
  "text": "This product is really good."
}
```

Exemple de réponse :

```json
{
  "prediction": 2,
  "label": "Positive",
  "probability": 0.94,
  "confiance": 0.94
}
```

Les labels retournés sont automatiquement convertis :

```text
0 → Negative
1 → Neutral
2 → Positive
```

---

## Exemple avec curl

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product is really good."
  }'
```

---

## Documentation FastAPI

Une fois l'API lancée :

```text
http://127.0.0.1:8000/docs
```

Documentation alternative ReDoc :

```text
http://127.0.0.1:8000/redoc
```

---

## Health Check

```bash
curl http://127.0.0.1:8000/health
```

---

## Interface Streamlit — Coming Soon

Une interface Streamlit sera ajoutée au projet afin de permettre une utilisation interactive du modèle.

Le fonctionnement prévu est simple :

```text
User text
    ↓
Streamlit
    ↓
FastAPI
    ↓
CNN Sentiment Model
    ↓
Negative / Neutral / Positive
```

L'utilisateur pourra saisir directement une phrase, par exemple :

```text
I really enjoyed this product.
```

L'application affichera ensuite le sentiment prédit :

```text
Positive
```

ainsi que les informations associées à la prédiction, notamment la probabilité ou le niveau de confiance du modèle.

L'objectif est de rendre le modèle testable sans avoir besoin d'utiliser directement l'API ou un notebook.

---

## Structure

```text
.
├── api/
│   ├── app.py                       # Application FastAPI
│   ├── models.py                    # Routes de prediction
│   ├── schemas.py                   # Schemas Pydantic
│   └── utils.py                     # Cleaning et fonctions de prediction
│
├── core/
│   └── config.py                    # Configuration applicative
│
│
├── models/
│   ├── cnn_tokenization.keras       # CNN principal
│   ├── full_cnn_tokenization.keras  # CNN principal
│   ├── cnn_word2vec.keras           # CNN avec Word2Vec
│   ├── tokenizer_a.pkl              # Tokenizer model a
│   └── tokenizer_b.pkl              # Tokenizer model b
│   └── tokenizer_full.pkl           # Tokenizer
│
├── notebooks/
│   └── notebook.ipynb               # EDA, preprocessing, training et comparaison
│
└── streamlit/
│   ├── streamlit_app.py             # Interface utilisateur
```

---

## Lancer l'API

Depuis la racine du projet :

```bash
uv run uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
```

ou selon la configuration du projet :

```bash
uv run python main.py
```

L'API est ensuite disponible sur :

```text
http://127.0.0.1:8000
```

---

## Configuration LLM

La comparaison avec le LLM utilise des variables d'environnement.

Exemple :

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_api_key
```

Ces variables sont principalement nécessaires pour les expérimentations LLM du notebook et ne sont pas requises pour effectuer une prédiction avec le CNN local.

---

## Modèle actuellement utilisé par l'API

L'API utilise actuellement :

```text
models/full_cnn_tokenization.keras
```

Il s'agit de la **Baseline A — CNN + Tokenization**, retenue face à la baseline Word2Vec après comparaison des performances de validation.

Le pipeline d'inférence est :

```text
Raw text
   ↓
Cleaning
   ↓
Tokenizer
   ↓
Padding — max length 200
   ↓
CNN
   ↓
Softmax probabilities
   ↓
argmax
   ↓
Negative / Neutral / Positive
```

---

## Limitations

Comme tout modèle NLP relativement compact, le CNN possède certaines limites.

Il peut notamment rencontrer des difficultés avec :

* les négations complexes ;
* le sarcasme ;
* l'ironie ;
* les phrases ambiguës ;
* les formulations très différentes des données d'entraînement ;
* les mots absents ou rares dans le vocabulaire appris.

Exemple potentiellement difficile :

```text
I am not happy about my purchase.
```

Ces limitations constituent également l'un des intérêts de la comparaison avec des modèles de langage plus généralistes.

---

## Stack

### Machine Learning & NLP

```text
TensorFlow / Keras
scikit-learn
NLTK
Gensim
Word2Vec
NumPy
Pandas
<!-- Sentence Transformers -->
```

### API

```text
FastAPI
Pydantic
```

### LLM

```text
OpenAI API
tiktoken
python-dotenv
```

### Interface

```text
Streamlit
```

### Experimentation

```text
Jupyter Notebook
Matplotlib
Seaborn
```

---

## Notes

* Le projet effectue une classification multiclasse : `Negative`, `Neutral` et `Positive`.
* Le CNN Tokenization est actuellement le modèle principal utilisé pour l'inférence.
* Le tokenizer et le modèle doivent être utilisés ensemble afin de reproduire le preprocessing réalisé pendant l'entraînement.
* La comparaison avec le LLM sert principalement de benchmark expérimental.
* La partie Sentence Transformers constitue la base d'une future fonctionnalité de recherche sémantique.
* Concernant la version anglaise, merci de prendre en compte de la limitation de CNN face au `negation handling`
