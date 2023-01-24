import streamlit as st
import sklearn
import joblib,os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random

# Loading Models
def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model

def main():
    html_templ = """<center><h2 style="", text-align="center">Prediksi Gaji Karyawan Berdasarkan Pengalaman dengan Linear Regression</h2></center>"""
    html_temp2 = """<center><i><p style="", text-align="center">Project Machine Learning - 3 TI E</p></i></center>"""
    st.markdown(html_templ,unsafe_allow_html=True)
    st.markdown(html_temp2,unsafe_allow_html=True)

    st.sidebar.image("img/logo_pcr.png",use_column_width=True)
    activity = ["Dashboard","Visualisasi Data"]
    
    choice = st.sidebar.selectbox("Menu",activity)
    
# Salary Determination CHOICE
    if choice == 'Visualisasi Data':
        work = st.radio(
            "Apakah Anda Seorang Pekerja ?",
            ('Ya', 'Tidak'))
        OHE_yes = 0

        if work == 'Ya':
            st.text('Anda Seorang Pekerja')
            OHE_yes= 1
        else:
            st.text("Anda Bukan Seorang Pekerja")

        experience = st.slider("Pilih Tahun Pengalaman Bekerja :",0,20)
        if st.button("Hitung Gaji Berdasarkan Tahun Pengalaman yang Dipilih"):

            regressor = load_prediction_model("model/linear_regression_salary.pkl")
            experience_reshaped = np.array(experience).reshape(-1,1)

            predicted_salary = regressor.predict(experience_reshaped)

            st.info("Gaji Terkait dengan {} Tahun Pengalaman Bekerja : {} dollar".format(experience,(predicted_salary[0][0].round(2))))
        spectra = st.file_uploader("Upload file : ", type={"csv", "txt"})
        if spectra is not None:
            spectra_df = pd.read_csv(spectra)
            st.write(spectra_df)

        st.text("Test Set Salary and Experience :")
        st.image("img/TestSet.png",use_column_width=True)
        st.text("Training Set Salary and Experience :")
        st.image("img/TrainingSet.png",use_column_width=True)

            

# About CHOICE
    if choice == 'Dashboard':
        st.subheader("")
        st.markdown("""
            ## Apa itu Linear Regression? :
            Regresi linear adalah teknik analisis data yang memprediksi nilai data yang tidak diketahui dengan menggunakan nilai data lain yang terkait dan diketahui. 
            Secara matematis memodelkan variabel yang tidak diketahui atau tergantung dan variabel yang dikenal atau independen sebagai persamaan linier.
            """)
        st.markdown(" ## Library yang Digunakan :")

        st.code("""
            # Libraries
            import streamlit as st
            import sklearn
            import joblib,os
            import numpy as np 
            import pandas as pd
            import matplotlib.pyplot as plt
            import random 
            """)
        st.markdown("""
            ## Anggota Kelompok :
            1. Satria Tofa Anugrah
            2. Soeltan Bima Manggala Sidi
            3. Vito Baihaki Afif
            """)

if __name__ == '__main__':
    main()
