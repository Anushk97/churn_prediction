
## Project Folder Structure
```bash
Telco Customer Churn Project/
│
├── data/
│   ├── churn_data_regulated.parquet
│   ├── churn_data.csv
│   ├── ...
│
├── model/
│   └── cat_model.cbm
│
├── src/
│   ├── fast-api.py
│   ├── predict.py
│   ├── streamlit-app.py
│   └── train_model.py
│
└── Dockerfile
└── requirements.txt
```

## Project Files

- **data/**: Contains the data files used for the project.
- **model/**: Contains the trained model file.
- **src/**: Contains the source code of the project. Model training, prediction and service/application development are located in this folder.

## Steps
1. **Data Preprocessing and Model Development**: Using the script `train_model.py`, Telco Customer data is preprocessed and a machine learning model is created using the CatBoost model.
2. **Interface**: Using the model, the `streamlit-app.py` script allows the user to enter new customer information and based on this information the churn probability is estimated. Furthermore, the overall SHAP graph of the model and the specific SHAP graph of the selected customer are shown.
3. **API**: The `fast-api.py` script creates an API using the model created with train_model.py. This API takes customer data and returns the churn probability.
4. **Automation**: Using Docker, the project is containerized and made executable through the `predict.py` script. This script takes customer data and calculates the churn probability.


## Usage

### 
1. Clone the project
```bash
git clone https://github.com/rolmez/Customer-Churn-Project.git
```
2. Go to the project directory
```bash
cd Customer-Churn-Project
```
3. Install dependencies
```bash
pip install requirements.txt
```
4. Go to the src directory
```bash
cd src
```
- For train model
```bash
python train_model.py
```
- For Streamlit app
```bash
streamlit run streamlit-app.py
```
- For API
```bash
python fast-api.py
```
Go to this address to access API: http://127.0.0.1:5000/docs

- For predict
```bash
python predict.py
```
### or Docker
1. Run the following command to create the Docker container in the project's home folder:
```bash
  docker build -t telco-churn .
```
2. Run the following command to start the Docker container:
```bash
  docker run -it telco-churn
```