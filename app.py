import streamlit as st
import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler


# loading the saved model
model=pickle.load(open('xg_model.pkl','rb'))
sc=pickle.load(open('sc_model.pkl','rb'))


# creating a function for Prediction

def academic_pred(input_data):

    input_data_np=np.array(input_data)
    input_data_reshaped = input_data_np.reshape(1,-1)

    input_data_sc=sc.transform(input_data_reshaped)
    prediction=model.predict(input_data_sc)
    
    print(prediction)

    if (prediction[0] == 0):
      return 'Dropout'
    
    elif (prediction[0] == 1):
       return 'Enrolled'
    else:
      return 'Graduate'


def main():
   
  # giving a title
  st.title('Academic Prediction Web App 📙🧾')
    

  # getting the input data from the user
  Marital_status=st.text_input('Marital Status')
  Application_mode=st.text_input('Application Mode')
  Application_order=st.text_input('Application Order')
  Course=st.text_input('Course')
  Daytime_evening_attendence=st.text_input('Daytime Evening Attendence')
  Previous_qualification=st.text_input('Previous Qualification')
  Previous_qualification_grade=st.text_input('Previous Qualification Grade')
  Nacionality=st.text_input('Nacionality')
  Mothers_qualification=st.text_input('Mothers Qualification')
  Fathers_qualification=st.text_input('Fathers Qualification')
  Mothers_occupation=st.text_input('Mothers Occupation')
  Fathers_occupation=st.text_input('Fathers Occupation')
  Admission_grade=st.text_input('Admission Grade')
  Displaced=st.text_input('Displaced')
  Educational_special_needs=st.text_input('Educational Special Needs')
  Debtor=st.text_input('Debtor')
  Tuition_fees_up_to_date=st.text_input('Tuition Fees Up to Date')
  Gender=st.text_input('Gender')
  Scholarship_holder=st.text_input('Scholarship Holder')
  Age_at_enrollment=st.text_input('Age at Enrollment')
  International=st.text_input('International')
  Curricular_units_1st_sem_credited=st.text_input('Curricular Units 1st Sem Credited')
  Curricular_units_1st_sem_enrolled=st.text_input('Curricular Units 1st Sem Enrolled')
  Curricular_units_1st_sem_evaluations=st.text_input('Curricular Units 1st Sem Evaluations')
  Curricular_units_1st_sem_approved=st.text_input('Curricular Units 1st Sem Approved')
  Curricular_units_1st_sem_grade=st.text_input('Curricular Units 1st Sem Grade')
  Curricular_units_1st_sem_without_evaluations=st.text_input('Curricular Units 1st Sem Without Evaluations')
  Curricular_units_2nd_sem_credited=st.text_input('Curricular Units 2nd Sem Credited')
  Curricular_units_2nd_sem_enrolled=st.text_input('Curricular Units 2nd Sem Enrolled')
  Curricular_units_2nd_sem_evaluations=st.text_input('Curricular Units 2nd Sem Evaluations')
  Curricular_units_2nd_sem_approved=st.text_input('Curricular Units 2nd Sem Approved')
  Curricular_units_2nd_sem_grade=st.text_input('Curricular Units 2nd Sem Grade')
  Curricular_units_2nd_sem_without_evaluations=st.text_input('Curricular Units 2nd Sem Without Evaluations')
  Unemployment_rate=st.text_input('Unemployment Rate')
  Inflation_rate=st.text_input('Inflation Rate')
  GDP=st.text_input('GDP')


  # Code for prediction
  academics=''


  # creating a button for Prediction 
  if st.button('Academic Test Result'):
     academics=academic_pred([Marital_status, Application_mode, Application_order, Course, 
                              Daytime_evening_attendence, Previous_qualification, 
                              Previous_qualification_grade, Nacionality, Mothers_qualification, 
                              Fathers_qualification, Mothers_occupation, Fathers_occupation, 
                              Admission_grade, Displaced, Educational_special_needs, Debtor, 
                              Tuition_fees_up_to_date,Gender, Scholarship_holder, Age_at_enrollment,
                              International, Curricular_units_1st_sem_credited,
                              Curricular_units_1st_sem_enrolled, Curricular_units_1st_sem_evaluations,
                              Curricular_units_1st_sem_approved, Curricular_units_1st_sem_grade,
                              Curricular_units_1st_sem_without_evaluations, Curricular_units_2nd_sem_credited,
                              Curricular_units_2nd_sem_enrolled, Curricular_units_2nd_sem_evaluations,
                              Curricular_units_2nd_sem_approved, Curricular_units_2nd_sem_grade,
                              Curricular_units_2nd_sem_without_evaluations,
                              Unemployment_rate, Inflation_rate, GDP])
     
  st.success(academics)





if __name__ == '__main__':
    main()