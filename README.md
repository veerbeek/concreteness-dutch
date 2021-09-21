# Concreteness
Method to estimate the concreteness of a word on a scale from 1 to 5 based on its fasttext embedding. Trained on the `cc.nl.300.bin` Dutch fasttext embeddings. Also works on unknown words.

### Dependencies:
* scikit-learn
* LightGBM

### Evaluation
|                              | $$R^2$$ | MAE  | $$r$$ (test) | $r$ (train) |
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


