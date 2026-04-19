# CFD + Machine Learning Flow Prediction

This project investigates the use of machine learning models to predict viscoplastic flow profiles and compares their performance against CFD results generated using the CIP scheme.

## Objective

To evaluate whether machine learning models can accurately reproduce flow depth distributions under varying yield stress conditions, and assess their generalization capability on unseen cases.

## Methodology

- Training data: Multiple yield stress cases (20–5000 Pa)
- Testing data: Unseen case (50 Pa)
- Input features:
  - Distance ratio (x/x₀)
  - Yield stress
- Output:
  - Flow depth ratio (h/h₀)
 
## Reference

The CFD data used in this project is based on the following study:

Tan, J. F., Puay, H. T., & Lau, S. C. (2025).  
*Parametric Sensitivity Analyses of Mudflow Rheology with Depth-Averaged Model*.  
Journal of Advanced Research in Fluid Mechanics and Thermal Sciences, 131(2), 78–94.

### Models used:
- Polynomial Regression (degree = 4)
- Random Forest Regressor

## Evaluation Metrics

Model performance is evaluated using both statistical and physics-based metrics:

### Statistical metrics:
- R² (coefficient of determination)
- RMSE (root mean square error)
- MAE (mean absolute error)

### Physics-based metrics:
- Flow front position
- Active region error (h/h₀ > 0.01)
- Area under curve (volume proxy)

## Results

- Random Forest significantly outperforms Polynomial Regression:
  - ~4× lower RMSE in the active flow region
  - ~5× lower MAE
- Both models capture the general flow profile
- Largest errors occur near the flow front due to sharp gradients

## Key Insight

Machine learning models can effectively approximate smooth flow regions, but struggle to accurately capture sharp physical features such as the flow front location.


## 👤 Author

Jeff Tan  
Civil Engineering + CFD → Transitioning into AI / Data-driven modeling



