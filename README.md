# MLP from Scratch na Base Íris: Depuração, Classificação das Três Espécies e Análise de Overfitting

**Dupla:**
- Gustavo Gabriel Ripka — GRR20203935
- Arthur Barreto Godoi — GRR20224377

## Visão geral

Estudo, depuração e experimentos com a implementação de **MLP (Multilayer
Perceptron) from scratch** apresentada em aula (adaptada do *notebook* de
Vítor Gama Lemos), na classificação das três espécies do dataset Íris
(150 amostras, 4 atributos, 3 classes equilibradas). Três eixos, conforme o
enunciado do Trabalho 5:

1. **Estudo e depuração** — cinco bugs encontrados e corrigidos no código
   original:
   - `predict()` não aplicava a função de ativação (rede avaliada como modelo
     linear);
   - erro acumulado nunca zerado entre épocas (MSE exibido crescia
     monotonicamente);
   - fator arbitrário `0.05` no erro e registro a cada 50 épocas;
   - typo `iasHiddenValue` e ausência de semente reprodutível;
   - **bug grave:** `random.shuffle` do Python sobre `ndarray` 2D do NumPy —
     as trocas viram duplicações por *aliasing* de *views*, perdendo amostras
     e quase eliminando a classe *virginica*. Era isso, e não o
     backpropagation (que estava correto), que impedia classificar as três
     classes.
2. **Classificação das três classes** — split estratificado 80/20
   (embaralhando índices); baseline corrigido atinge **96,7% de teste**;
   grade sistemática (ativação × neurônios ocultos × learning rate, 3
   sementes, z-score); melhor configuração (**tanh, 5 neurônios, lr 0,01**)
   alcança **97,8% ± 1,9** e, com 1500 épocas, **100% nas três classes**
   (matriz de confusão perfeita).
3. **Convergência, erros e overfitting** — curvas de MSE por função de
   ativação (sigmoid atinge piso menor que tanh com alvos one-hot; ReLU morre
   precocemente com inicialização U(−1,1) e bias −1); normalização z-score
   exige recalibrar o learning rate (0,005 → 0,01–0,05); gap treino–teste
   cresce com o tamanho da camada oculta (`results/overfitting.csv`).

## Resultados principais

| Experimento | Resultado |
|---|---|
| Baseline corrigido (sigmoid, h=5, lr 0,005, 3000 épocas, atributos crus) | 96,7% teste / MSE 0,44 → 0,026 |
| Grade (z-score, 3 sementes): melhor célula (tanh, h=5, lr 0,01) | 97,8% ± 1,9 teste |
| Melhor configuração + 1500 épocas | **100% no teste (3/3 classes)** |
| Overfitting (gap treino–teste, h=2→20) | −0,8 a +5,3 p.p. |

Tabelas completas em `results/` (`grid_bruto.csv`, `grid_resumo.csv`,
`overfitting.csv`) e no `mlp_iris_scratch.ipynb` executado.

## Como rodar

O *notebook* final é gerado a partir das fontes em `cells.py` e executado com
`nbclient`:

```bash
python3 build_run.py "$(pwd)" 4
```

O argumento `4` inclui todas as células (grade, convergência, matriz de
confusão e overfitting); estágios menores geram versões parciais. O relatório
está em `relatorio.tex` (PDF já compilado em `relatorio.pdf`).

Dependências (Python 3): `numpy`, `pandas`, `scikit-learn`, `matplotlib`,
`seaborn`, `nbformat`, `nbclient`.

## Estrutura de pastas

```
mlp-scratch-iris/
├── notebook_original.ipynb   # implementação da aula (referência, com os bugs)
├── cells.py                  # fontes das células do notebook final
├── build_run.py              # gera e executa o notebook via nbclient
├── mlp_iris_scratch.ipynb    # notebook final executado (com outputs)
├── results/                  # CSVs: grade de experimentos e overfitting
├── relatorio.tex             # relatório do trabalho (IEEEtran)
└── relatorio.pdf             # relatório compilado
```
