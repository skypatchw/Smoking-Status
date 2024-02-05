# Smoking Status Prediction

Smoking adversely affects multiple organs of the body and is the leading cause of preventable mortality worldwide. According to the World Health Organisation, cigarette smoking kills over 8 million people globally each year. Although smoking cessation strategies have been advocated, their success remains limited, with traditional counseling often considered ineffective and time-consuming.

To address this, a solution has been proposed that entails developing predictive models using machine learning techniques to identify those individuals who have higher chances of quitting - based on certain physiological and biochemical factors. This promising approach has the potential to enhance the effectiveness of smoking cessation interventions and ultimately improve public health outcomes.

This repository supplies the code of developing a Machine Learning (ML) model that predicts whether an adult, residing in Korea, is a smoker or a non-smoker.  

## Implementation

**ML Model Type**: Supervised

**Algorithms**:

-	Xtreme Gradient Boost (78.5% AUC)

**Hyperparameter Tuning Tool**: Optuna 

## Quick Start





## Requirements

Python version 3.10.7

Python libraries: pandas, numpy, sklearn, matplotlib, seaborn, warnings, xgboost, dtreeviz 

## Data

All necessary data can be found in the "Files" folder of the repository or on Kaggle [https://www.kaggle.com/competitions/playground-series-s3e24/data]

## Additional Information

The dataset comprises of 159,256 samples that were synthetically generated using patient records originally compiled by the National Health Insurance Corporation of Korea from a sub-population residing there. 
The training set contains data regarding bio-markers of 268 smokers and 500 non-smokers. This information is contained in the column named 'Smoking'. Only participants whose age was at least 21 years were included.

### Feature Descriptions

| Name | Description |
| ---- | ----------- |
| **Age** | Age of patient, **grouped by 5-year increments** |
| **Height** | Height of patient, **grouped by 5-cm increments** |
| **Weight** | Weight of patient, **grouped by 5-kg increments** |
| **Waist** | Waist circumference |
| **Eyesight (left)** | Visual acuity in left eye |
| **Eyesight (right)** | Visual acuity in right eye |
| **Hearing (left)** | Hearing in left ear |
| **Hearing (right)** | Hearing in right ear |
| **Systolic** | Blood pressure when the heart contracts |
| **Relaxation** | Blood pressure (diastolic) when the heart relaxes |
| **Fasting Blood Sugar** | Blood sugar level in fasted state |
| **Cholesterol** | Amount of cholesterol in the blood|
| **Triglyceride** | Amount of triglycerides in blood |
| **HDL** | High Density Lipoprotein - higher levels protect against heart disease |
| **LDL** | Low Density Lipoprotein - higher levels raise risk for heart disease |
| **Hemoglobin** | Amount of hemoglobin in the blood |
| **Urine Protein** | Amount of protein present in urine |
| **Serum Creatinine** | Creatinine level in the blood |
| **AST** | Aspartate transaminase (liver enzyme) – higher blood levels indicate liver damage |
| **ALT** | Alanine transaminase (liver enzyme) – higher blood levels indicate liver damage |
| **GTP** | Gamma-glutamyl transpeptidase (liver enzyme) – higher blood levels indicate liver damage |
| **Dental Caries** | 0=absent, 1=present |
| **Smoking** | 0=non-smoker, 1=smoker |


## Citation

The data was generated from the original data compiled by the National Insurance Corporation of Korea: https://www.data.go.kr/data/15007122/fileData.do#/tab-layer-file

**Review on Tobacco Usage**

Samet JM. Tobacco smoking: the leading cause of preventable disease worldwide. Thorac Surg Clin. 2013 May;23(2):103-12. doi: 10.1016/j.thorsurg.2013.01.009. Epub 2013 Feb 13. PMID: 23566962.
