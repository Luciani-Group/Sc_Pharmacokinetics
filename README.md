# Towards Animal-Free Approaches in Subcutaneous Drug Delivery: a Pain Management Case Study

<img src="https://img.shields.io/badge/Python-3.11.9-blue?style=flat-square"/> <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>

Subcutaneous (s.c.) administration offers practical advantages for long-acting drug delivery, yet its complex tissue environment, lack of standardized models, and limited regulatory guidance pose challenges for predicting pharmacokinetics (PK). Here, we present a complementary set of experimental and computational approaches to reduce reliance on animal testing while maintaining translational relevance. Ex vivo skin models captured the influence of tissue variability, while 3D-printed constructs and molded agarose hydrogels enabled standardized, reproducible in vitro drug release studies. In parallel, machine learning models trained on curated rodent datasets predicted key PK parameters with strong agreement to in vivo data, providing an alternative that bypasses additional animal experiments and resource-intensive assays. Finally, a liposomal carprofen formulation case study demonstrated the translational potential of these methods in veterinary applications. Together, these strategies illustrate routes toward animal-free drug development: simplified experimental systems that mimic s.c. release and computational models that predict systemic PK. By combining them conceptually, we outline a framework that advances the 3Rs principles while supporting mechanistic and predictive understanding of s.c. drug delivery. 


## Environment

To ensure reproducibility and ease of setup, the project was developed using Python 3.11.9 within an Anaconda 24.3.0 (https://www.anaconda.com/) environment.

## Dependencies

The project uses the following Python packages and libraries:

### Data manipulation and analysis
pandas (v. 2.2.2),
numpy (v. 1.26.4),
rdkit (v. 2024.03.2)

### Machine learning models and tools
xgboost (v. 2.0.3),
scikit-learn (v. 1.4.2)

### Visualization
plotly (v. 5.22.0),
seaborn (v. 0.12.2),
shap (0.45.1)



