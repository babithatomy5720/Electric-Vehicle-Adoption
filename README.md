# Electric Vehicle Adoption & CAFV Eligibility Prediction

## 📌 Project Overview

The **Electric Vehicle Adoption & CAFV Eligibility Prediction** project is an end-to-end data science and analytics solution built using electric vehicle registration data.

The project analyzes electric vehicle adoption patterns across manufacturers, vehicle models, geographic regions, vehicle types, electric range, and CAFV eligibility.

It combines **data analysis, exploratory data analysis, feature engineering, machine learning, Power BI visualization, and Streamlit deployment** into a complete workflow.

The project follows the complete data science lifecycle:

**Data Profiling → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Machine Learning → Model Optimization → Business Intelligence → Model Deployment**

---

## 🎯 Business Problem

The increasing adoption of electric vehicles generates large volumes of registration data containing information about vehicle characteristics, manufacturers, locations, electric range, and CAFV eligibility.

The objective of this project is to transform this raw registration data into actionable insights and a predictive machine learning solution.

The project addresses the following questions:

- How are electric vehicles distributed across different regions?
- Which manufacturers and models dominate EV registrations?
- What is the distribution between BEVs and PHEVs?
- How is electric range distributed across vehicles?
- What proportion of vehicles are CAFV eligible?
- Which vehicle characteristics are most associated with CAFV eligibility?
- Can machine learning be used to predict CAFV eligibility for labeled vehicle records?

---

# 🎯 Project Objectives

### Data Analytics

- Profile the raw electric vehicle registration dataset.
- Identify missing values and data quality issues.
- Clean and standardize the dataset.
- Analyze EV adoption patterns.
- Analyze manufacturers and vehicle models.
- Analyze geographic distribution.
- Analyze electric range.
- Analyze vehicle age and registration periods.
- Analyze CAFV eligibility.

### Machine Learning

- Prepare a model-ready dataset.
- Engineer relevant vehicle and geographic features.
- Build classification models.
- Compare Logistic Regression and Random Forest.
- Tune the Random Forest model using GridSearchCV.
- Evaluate model performance.
- Analyze feature importance.
- Perform sensitivity analysis by removing Electric Range.
- Save the final production model.

### Visualization and Deployment

- Build an interactive Power BI dashboard.
- Build a Streamlit prediction application.
- Deploy the trained machine learning model for interactive predictions.

---

# 📊 Dataset

The project uses an **Electric Vehicle Population dataset** containing vehicle registration records.

### Original Dataset

- **Rows:** 270,262
- **Columns:** 16

The dataset contains information about registered electric vehicles, including their:

- Location
- Manufacturer
- Model
- Model year
- EV type
- Electric range
- CAFV eligibility
- Geographic coordinates
- Electric utility
- Registration information

The original dataset was first profiled and then cleaned before being used for analysis and machine learning.

---

# 📋 Dataset Description

The cleaned dataset contains **270,262 rows and 27 columns**.

| Column | Description |
|---|---|
| `VIN_1-10` | Partial Vehicle Identification Number |
| `County` | County where the vehicle is registered |
| `City` | City where the vehicle is registered |
| `State` | State of vehicle registration |
| `Postal_Code` | Postal code of the registration location |
| `Model_Year` | Model year of the vehicle |
| `Make` | Vehicle manufacturer |
| `Model` | Vehicle model |
| `Electric_Vehicle_Type` | Type of electric vehicle, such as BEV or PHEV |
| `Clean_Alternative_Fuel_Vehicle_CAFV_Eligibility` | Original CAFV eligibility information |
| `Electric_Range` | Electric driving range of the vehicle |
| `Legislative_District` | Legislative district associated with the registration |
| `DOL_Vehicle_ID` | Department of Licensing vehicle identifier |
| `Vehicle_Location` | Geographic location information |
| `Electric_Utility` | Electric utility associated with the vehicle location |
| `2020_Census_Tract` | 2020 Census tract |
| `Electric_Range_Available` | Indicates whether electric range information is available |
| `Electric_Range_Bucket` | Categorized electric range |
| `CAFV_Status` | Standardized CAFV eligibility category used for analysis |
| `Longitude` | Geographic longitude |
| `Latitude` | Geographic latitude |
| `Vehicle_Age` | Calculated vehicle age |
| `Vehicle_Vintage` | Categorized vehicle age |
| `Premium_Brand` | Indicator for premium vehicle brands |
| `Is_BEV` | Indicator identifying Battery Electric Vehicles |
| `Registration_Era` | Categorized registration period |
| `Range_Efficiency` | Categorized electric range availability/level |

---

# 🔎 Data Profiling

The first stage of the project was **data profiling**.

The dataset was examined to understand:

- Dataset dimensions
- Column names
- Data types
- Missing values
- Unique values
- Duplicate records
- Numerical distributions
- Categorical distributions
- Potential inconsistencies

The profiling stage helped identify the important variables and data quality issues before moving into cleaning and analysis.

---

# 🧹 Data Cleaning

The data cleaning stage prepared the raw dataset for analysis.

The cleaning process included:

- Handling missing values.
- Standardizing categorical values.
- Converting columns to appropriate data types.
- Processing electric range values.
- Standardizing CAFV eligibility categories.
- Preparing geographic information.
- Removing unnecessary fields from the machine learning dataset.
- Creating additional analytical features.

The cleaned dataset was saved as:

`ev_cleaned_dataset.csv`

---

# 📈 Exploratory Data Analysis

Exploratory Data Analysis was performed to understand EV adoption patterns and identify important relationships in the dataset.

The analysis covered:

- EV type distribution
- Manufacturer distribution
- Model distribution
- Geographic distribution
- Electric range
- CAFV eligibility
- Vehicle age
- Premium brands
- Registration era
- Range availability

---

# 🚗 EV Type Distribution

The dataset contains two major EV types:

| EV Type | Records |
|---|---:|
| BEV | 215,859 |
| PHEV | 54,403 |

Battery Electric Vehicles represent the majority of the registered vehicles in the dataset.

This indicates that fully electric vehicles form the dominant EV category within the analyzed registration population.

---

# 🏭 Manufacturer Analysis

Manufacturer-level analysis was performed to identify brands with the highest number of EV registrations.

### Key Finding

Tesla is the leading manufacturer in the dataset, with approximately:

**111,049 registrations**

This indicates a strong representation of Tesla vehicles in the EV registration population.

---

# 🚘 Model Analysis

Vehicle model analysis was performed to identify the most frequently registered EV models.

### Key Finding

The Tesla Model Y is one of the most represented models, with approximately:

**57,335 registrations**

This highlights the strong presence of high-volume EV models in the dataset.

---

# 🗺️ Geographic Analysis

Geographic analysis was performed using:

- County
- City
- Latitude
- Longitude

### Key Findings

**King County** has approximately:

**133,903 registrations**

**Seattle** has approximately:

**42,125 registrations**

The distribution shows that EV registrations are strongly represented in major urban and metropolitan areas.

---

# 🔋 Electric Range Analysis

Electric range was analyzed using:

- Mean
- Standard deviation
- Minimum and maximum values
- Range availability
- Range buckets

For records with available electric range information:

- **Mean:** approximately 40.39
- **Standard deviation:** approximately 79.34

A major characteristic of the dataset is the large number of records with zero electric range.

Approximately:

**169,872 records (62.85%)**

have zero electric range.

This is an important data characteristic because zero range may represent vehicles for which range is not applicable or unavailable rather than vehicles literally having zero driving range.

---

# 📊 Electric Range Buckets

Electric range was categorized into the following groups:

| Range Bucket | Records |
|---|---:|
| No Range | 169,877 |
| Short | 50,183 |
| Medium | 16,304 |
| Long | 33,898 |

These categories were created to make range-related analysis easier and more interpretable.

---

# 📋 CAFV Eligibility Analysis

The original CAFV eligibility information was standardized into three categories:

- Eligible
- Not Eligible
- Unknown

The standardized target variable is:

`CAFV_Status`

The encoding used for analysis and machine learning is:

| CAFV Status | Encoded Value |
|---|---:|
| Eligible | 0 |
| Not Eligible | 1 |
| Unknown | 2 |

For binary machine learning, the `Unknown` category was excluded.

Therefore, the machine learning problem focuses on:

**Eligible vs Not Eligible**

---

# 🚙 Vehicle Age Analysis

Vehicle age was calculated and then categorized into vehicle vintage groups.

The resulting categories were:

| Vehicle Vintage | Records |
|---|---:|
| Recent | 109,574 |
| New | 96,471 |
| Mid-Age | 50,515 |
| Older | 13,702 |

The following encoding was used in the machine learning pipeline:

| Vehicle Vintage | Encoded Value |
|---|---:|
| Mid-Age | 0 |
| New | 1 |
| Older | 2 |
| Recent | 3 |

---

# ⭐ Premium Brand Analysis

A `Premium_Brand` feature was created to identify premium vehicle brands.

The distribution was:

| Category | Records |
|---|---:|
| Premium Brand | 140,212 |
| Non-Premium Brand | 130,050 |

This feature was included as one of the predictors in the machine learning model.

---

# 📅 Registration Era Analysis

Registration years were grouped into meaningful adoption periods.

The resulting distribution was:

| Registration Era | Records |
|---|---:|
| Modern EV Era | 155,795 |
| Expansion Phase | 62,349 |
| Growth Phase | 42,846 |
| Early Adoption | 9,272 |

This feature provides a higher-level view of the evolution of EV registrations over time.

---

# 📊 Range Efficiency Analysis

A `Range_Efficiency` feature was created to categorize range availability and level.

The distribution was:

| Range Efficiency | Records |
|---|---:|
| Unknown | 169,877 |
| Low | 50,182 |
| High | 32,480 |
| Moderate | 17,723 |

---

# 🧮 Feature Engineering

Feature engineering was performed to create meaningful variables for both analytics and machine learning.

The engineered features include:

- `Electric_Range_Available`
- `Electric_Range_Bucket`
- `CAFV_Status`
- `Vehicle_Age`
- `Vehicle_Vintage`
- `Premium_Brand`
- `Is_BEV`
- `Registration_Era`
- `Range_Efficiency`

The final ML dataset was then created using the selected predictive features.

---

# 🤖 Machine Learning

The machine learning objective is to predict CAFV eligibility for vehicles with known CAFV labels.

The final modeling dataset contains:

**100,390 records**

The target variable is:

`CAFV_Status`

Only the following classes were used:

- `0` → Eligible
- `1` → Not Eligible

Records with:

`CAFV_Status = 2`

were excluded from binary classification because they represent Unknown eligibility.

---

# 🎯 Target Distribution

The labeled dataset was divided into training and testing sets using a stratified split.

### Training Dataset

| Class | Records |
|---|---:|
| Eligible | 61,088 |
| Not Eligible | 19,224 |

### Testing Dataset

| Class | Records |
|---|---:|
| Eligible | 15,272 |
| Not Eligible | 4,806 |

The stratified split preserves the class distribution between training and testing datasets.

---

# 🧩 Final Machine Learning Features

The final Random Forest model uses 12 features:

```text
County
City
Make
Model
Electric_Vehicle_Type
Electric_Range
Longitude
Latitude
Vehicle_Age
Vehicle_Vintage
Premium_Brand
Is_BEV
