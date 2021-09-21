# concreteness
Models to estimate the concreteness of a word on a scale from 1 to 5 based on its fasttext embedding. Trained on the `cc.nl.300.bin` Dutch fasttext embeddings. Only use on nouns, verbs and adjectives; also works on unknown words. 

Repository contains two models: `svr` and `lgbm`. The `svr` is slightly more accurate but relatively slow, `lgbm` is less accurate but faster.

### Dependencies:
* scikit-learn
* LightGBM

### Evaluation
|                              | R2 | MAE  | r (test) | r (train) |
|------------------------------|---------|------|--------------|-------------|
| SVR                          | 0.74    | 0.39 | 0.86         | 0.95        |
| LGBM                         | 0.69    | 0.44 | 0.83         | 0.89        |
| LR (Thompson & Lupyan, 2018) |         |      |              | 0.8         |

### Example usage

```python
from concreteness import WordConcreteness

wc = WordConcreteness(model='svr')

wc.score('boek')
## 4.83339

wc.score('coronavirus')
## 3.08055

wc.score('ideologie')
## 1.4905
```

Or, if you don't want bother with this tiny wrapper, just use the models directly:

```python
import pickle
import fasttext

svr = pickle.load(open('models/svr.p', 'rb'))
embeddings =  fasttext.load_model('cc.nl.300.bin')

word = 'boek'
word_embedding = embeddings.get_word_vector(word)

prediction = svr.predict([word_embedding])[0]
prediction
## 4.83339
```


