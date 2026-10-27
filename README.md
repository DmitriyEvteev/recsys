# Рекомендательные системы

**Студент:** Евтеев Дмитрий

| Задание | Папка | Описание |
|---|---|---|
| ДЗ 1 | [`hw1/`](hw1/) | пять моделей (ItemKNN, iALS, EASE, FM, iALS с кастомным таргетом) на Yambda-50M; данные - [`hw1/DATA.md`](hw1/DATA.md), эксперименты - [`hw1/experiments.ipynb`](hw1/experiments.ipynb) |

## Запуск ДЗ 1

```bash
cd hw1
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb
```
