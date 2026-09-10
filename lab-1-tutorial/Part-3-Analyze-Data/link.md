# Healthcare Dataset

**Source:** [Healthcare Dataset on Kaggle](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)

## About the Dataset

### Context

This synthetic healthcare dataset was created as a resource for data science, machine learning, and data analysis enthusiasts. It mimics real-world healthcare data, allowing users to practice, develop, and showcase data-manipulation and analysis skills in a healthcare context.

### Inspiration

The dataset addresses the need for practical, diverse healthcare data for educational and research purposes. Because real healthcare data are often sensitive and subject to privacy regulations, they can be difficult to access for learning and experimentation.

The data were generated with Python's Faker library to mirror the structure and attributes commonly found in healthcare records. By providing synthetic data, the creator aims to foster innovation, learning, and knowledge sharing in healthcare analytics.

## Dataset Information

Each column describes the patient, admission, or healthcare services provided, making the dataset suitable for a variety of healthcare data-analysis and modeling tasks.

| Column | Description |
| --- | --- |
| **Name** | Name of the patient associated with the healthcare record. |
| **Age** | Patient's age at admission, in years. |
| **Gender** | Patient's gender: `Male` or `Female`. |
| **Blood Type** | Patient's blood type, such as `A+` or `O-`. |
| **Medical Condition** | Primary medical condition or diagnosis, such as `Diabetes`, `Hypertension`, or `Asthma`. |
| **Date of Admission** | Date the patient was admitted to the healthcare facility. |
| **Doctor** | Doctor responsible for the patient's care during admission. |
| **Hospital** | Healthcare facility or hospital where the patient was admitted. |
| **Insurance Provider** | Patient's insurance provider, such as `Aetna`, `Blue Cross`, `Cigna`, `UnitedHealthcare`, or `Medicare`. |
| **Billing Amount** | Amount billed for healthcare services during admission, expressed as a floating-point number. |
| **Room Number** | Room number where the patient was accommodated. |
| **Admission Type** | Admission type: `Emergency`, `Elective`, or `Urgent`. |
| **Discharge Date** | Date the patient was discharged, calculated from the admission date and a realistic random length of stay. |
| **Medication** | Medication prescribed or administered during admission, such as `Aspirin`, `Ibuprofen`, `Penicillin`, `Paracetamol`, or `Lipitor`. |
| **Test Results** | Medical-test result: `Normal`, `Abnormal`, or `Inconclusive`. |

## Potential Uses

This dataset can be used to:

- Develop and test healthcare predictive models.
- Practice data cleaning, transformation, and analysis techniques.
- Create visualizations that identify healthcare trends.
- Learn and teach data science and machine learning concepts in a healthcare context.
- Frame **Test Results** as a multiclass classification target with the categories `Normal`, `Abnormal`, and `Inconclusive`.

## Privacy and Acknowledgment

This dataset is entirely synthetic. It contains no real patient information and does not violate privacy regulations. It is intended to support data science and healthcare analytics learning, exploration, and sharing within the Kaggle community.

## Image Credit

Image by BC Y from Pixabay.
