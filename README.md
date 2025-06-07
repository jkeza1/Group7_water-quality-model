# Water Quality Classification Model

A deep learning project to predict water potability using neural networks. Built by Group 7 as part of the Introduction to Machine Learning course.

## Project Structure

```
├── data/
│   └── water_potability.csv  
├── notebooks/
│   ├── excel_asaph_copy_of_formative_II_starter_code_.ipynb
│   ├── NICOLAS_Copy_Of_Formative_II_Starter_Code_.ipynb
│   ├── keza_joan_of_formative_ii_starter_code.ipynb
│   ├── nicolle_marizani_formative_II_starter_code_.ipynb
│   └── John_Ongeri_Ouma_formative_II_starter_code_.ipynb
├── report/
│   ├── gitlog.txt               
│   ├── report.md                
│   └── report.pdf              
└── README.md                    
```

## Team Members & Contributions

1. **Excel Asaph**
   - Model Architecture: 3 hidden layers (32-64-128)
   - Regularization: L1 (λ=0.01)
   - Dropout: 0.25
   - Results: 71% accuracy, F1=0.675
   - [View Notebook](notebooks/excel_asaph_copy_of_formative_II_starter_code_.ipynb)

2. **John Ongeri Ouma**
   - Progressive Dropout Implementation
   - Best Performance: 92.8% accuracy
   - F1 Score: 0.939
   - [View Notebook](notebooks/John_Ongeri_Ouma_formative_II_starter_code_.ipynb)

3. **Nicolas Muhigi**
   - L2 Regularization (λ=0.002)
   - Nadam Optimizer
   - Accuracy: 74.3%
   - F1 Score: 0.717

4. **Joan Keza**
   - L1 Regularization (λ=0.001)
   - Adagrad Optimizer
   - Accuracy: 66.7%
   - F1 Score: 0.390

5. **Nicolle Marizani**
   - RMSprop Optimizer
   - Accuracy: 65.4%
   - Precision: 0.750
   - Recall: 0.172

## Dataset

The dataset comes from [Kaggle's Water Quality and Potability Dataset](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability?select=water_potability.csv), and can be found here [water_potability.csv](data/water_potability.csv). It contains water quality measurements used to determine if water samples are safe for human consumption including:
- ph
- Hardness
- Solids
- Chloramines
- Sulfate
- Conductivity
- Organic_carbon
- Trihalomethanes
- Turbidity
- Potability (target variable)

## Model Comparisons

Detailed performance analysis and model comparisons can be found in our [comprehensive report](report/report.md).

### Key Metrics Overview

| Member | Accuracy | F1 Score | Recall | Precision |
|--------|----------|-----------|---------|------------|
| John | 92.8% | 0.939 | 0.929 | 0.948 |
| Nicolas | 74.3% | 0.717 | 0.702 | 0.735 |
| Excel | 71.0% | 0.675 | 0.701 | 0.692 |
| Joan | 66.7% | 0.390 | 0.622 | 0.554 |
| Nicolle | 65.4% | 0.280 | 0.172 | 0.750 |

## Key Findings

- Progressive dropout (John's approach) showed superior performance
- L2 regularization performed better than L1 for this dataset
- Early stopping on validation loss proved more reliable than accuracy
- Balanced approach to regularization and optimization was crucial

## Running the Models

1. Install required dependencies:
```bash
pip install tensorflow pandas numpy sklearn matplotlib jupyter
```

2. Open the Jupyter notebooks in the [notebooks](notebooks/) directory
3. Run cells sequentially to reproduce results

## Documentation

- Full analysis and methodology: [report.md](report/report.md)
- PDF version: [report.pdf](report/report.pdf)
- Development history: [gitlog.txt](report/gitlog.txt)