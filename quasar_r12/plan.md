# Quasar R1.2 Training Recipe

Quasar R1.2 is the routing behavioral model, with 1 OD pair (Madrid-London), 2 factors:
- Wind at cruise altitude.
- Link preference (unique to airline and OD pair).

## Key objectives
- Correctness of algorithm.
- Computational complexity.
- Log-likelihood as loss function.
- Accuracy compared to a held out test set:
  - Random test set (statistical generalization).
  - Autoregressive test set (predictive generalization).
- Interpretable features.

## Learning Algorithm
Gradient function:
> $$\nabla_\theta L_\theta = \frac{1}{|\mathcal{D}_s(i)|}\sum_{\sigma_{s\to i}} \Phi(\sigma_{s \to i}) +
> \frac{1}{|\mathcal{D}_m(ij)|}\sum_{\sigma_{i\to j}} \Phi(\sigma_{i \to j}) + \\
> \frac{1}{|\mathcal{D}_g(j)|}\sum_{\sigma_{j\to g}} \Phi(\sigma_{j \to g}) = \mathbb{E}_{\xi \sim \mathbb{P}(\cdot | \theta)} [\Phi(\xi)]$$

### Step by step learning algorithm
<font color='green'>*To be completed*</font>

## Execution Plan
- Label the segments: for each segment like `LFPG MIREX SAMER BUB PROPA VEGEA` we label this is a an `S` segment (starting from *CDG*).
- We then consider all OD pairs (maybe get the top 500 routes from IATA database, or infer from the `S` and `G` segments). These will be amortized in a neural network later so don't worry if we don't fully vex all the possibilities.
- We pick one OD pair (here: **Madrid-London**). We find the relevant segments from the label set and perform learning.
- Find the most effective way to save the data (since we might have lots of weight parameters, *for each OD pair, for each airline*).
- Documentation.

## Diary
| Date | Remark |
| --- | --- |
| 18/04/2025 | First draft of the plan | 

## Notes
- The number of total movements for some airports appear off: especially Paris and Frankfurt (53% and 64%). Could be a coverage problem?

| Airport                    | Code | 2024 Movements | Daily Avg | Obs | Coverage |
|----------------------------|:----:|---------------:|-----------------:|----------:|---------:|
| London Heathrow            | EGLL |      473,965   |   1,299         |     1,092 |   84%    |
| Amsterdam Schiphol         | EHAM |      473,815   |   1,298         |     1,051 |   81%    |
| Madrid-Barajas             | LEMD |      420,182   |   1,152         |       931 |   81%    |
| Istanbul Airport           | LTFM |      517,285   |   1,417         |       901 |   64%    |
| Paris-Charles de Gaulle    | LFPG |      460,916   |   1,262         |       664 |   53%    |
| Frankfurt Main             | EDDF |      430,436   |   1,179         |       662 |   56%    |
| Palma de Mallorca          | LEPA |      243,200   |     667         |       592 |   89%    |
| Lisbon Humberto Delgado    | LPPT |      225,268   |     617         |       583 |   94%    |
| Dublin Airport             | EIDW |      226,181   |     620         |       565 |   91%    |
| Sabiha Gökçen              | LTFJ |      241,536   |     662         |       529 |   80%    |