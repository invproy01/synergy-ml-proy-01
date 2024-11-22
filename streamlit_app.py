import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('models/kart_hp.pkl', 'rb'))

# Título de la aplicación
st.title("Aprendizaje automático en el diagnóstico de enfermedades cardiovasculares")

# Comprobar si el usuario ya presionó el botón "Empezar"
if 'started' not in st.session_state:
    st.session_state.started = False

# Crear un contenedor para la parte izquierda (imagen) y derecha (cuadro de texto)
col1, col2 = st.columns([1, 2])  # Esto divide la pantalla en 2 columnas

# Si no se ha presionado el botón "Empezar"
if not st.session_state.started:
    # Columna de la izquierda: Imagen
    with col1:
        st.image("cartoon-heart.png", use_container_width=True)

    # Columna de la derecha: Cuadro de texto y botón
    with col2:
        st.write("### Bienvenido! 💖")
        st.write(
            "Lastimosamente, las enfermedades que afectan al corazón y vasos sanguíneos constituyen una de las principales razones de muerte en el mundo."
            "Por este motivo, usaremos el Aprendizaje automático como una herramienta que ayude a la detección"
            "temprana y prevención de estas enfermedades facilitando el tratamiento oportuno de quien las" 
            "padece para evitar mayores complicaciones."
        
        )

        st.session_state.started = True

        # Botón "Empezar" en la parte inferior
        if st.button('Empezar'):
            st.session_state.started = True  # Cambiar el estado para mostrar los sliders
else:
    # Si ya se presionó el botón, se ocultan la imagen y el cuadro de texto, y se muestran los sliders

    with col1:
        st.empty()  # Limpiar la columna de la imagen
    
    with col2:
        st.empty()  # Limpiar la columna del cuadro de texto

    tab1, tab2 = st.tabs(['Formulario', 'Cargar datos'])
    with tab1: 
        # Título
        st.write("### Formulario de Parámetros de Salud")
        st.write("A continuación rellene el siguiente formulario con la información necesaria:")

        # Get the feature input from the user
        def get_user_input():
            # Edad
            edad = st.slider("Edad", min_value=20, max_value=80, value=30)

            # Sexo
            sexo = st.radio("Sexo", options=["Hombre", "Mujer"], index=0)
            sexo = 1 if sexo == "Hombre" else 0

            # Dolor en el pecho
            dolor_pecho = st.radio("Dolor en el pecho", options=["Angina típica", "Angina atípica", "Dolor tipo angina no cardiaca", "Ningún dolor"])
            dolor_pecho = {"Angina típica": 0, "Angina atípica": 1, "Dolor tipo angina no cardiaca": 1, "Ningún dolor": 3}[dolor_pecho]

            # Colesterol sérico
            restingBP = st.slider("Presión arterial en reposo en mmHg", min_value=94, max_value=200, value=120)

            # Colesterol sérico
            colesterol = st.slider("Colesterol sérico en mg/dl", min_value=126, max_value=564, value=200)

            # Azúcar en sangre en ayunas
            azucar_ayunas = st.checkbox("Azúcar en sangre > 120 mg/dl")

            # Electrocardiograma en reposo
            ecg_reposo = st.selectbox("Electrocardiograma en reposo", options=[0, 1, 2])

            # Frecuencia cardiaca máxima
            fc_max = st.slider("Frecuencia cardiaca máxima", min_value=71, max_value=202, value=150)

            # Angina producida por el ejercicio
            angina_ejercicio = st.radio("Angina producida por el ejercicio", options=["Sí", "No"])
            angina_ejercicio = 1 if angina_ejercicio == "Sí" else 0

            # Oldpeak
            oldpeak = st.slider("Oldpeak", min_value=0.0, max_value=6.2, value=1.0, step=0.1)

            # Pendiente del segmento ST
            pendiente_st = st.radio("Pendiente del segmento ST durante el ejercicio", options=["Ascendente", "Plana", "Descendente"])
            pendiente_st = {"Ascendente": 1, "Plana": 2, "Descendente": 3}[pendiente_st]

            # Número de vasos obstruidos
            vasos_obstruidos = st.selectbox("Número de vasos principales con obstrucción", options=[0, 1, 2, 3])

            # Store a dictionary into a variable
            datos = {
                'age': edad,
                'gender': sexo,
                'chestpain': dolor_pecho,
                'restingBP': restingBP,
                'serumcholestrol': colesterol,
                'fastingbloodsugar': azucar_ayunas,
                'restingrelectro': ecg_reposo,
                'maxheartrate': fc_max,
                'exerciseangia': angina_ejercicio,
                'oldpeak': oldpeak,
                'slope': pendiente_st,
                'noofmajorvessels': vasos_obstruidos
            }

            return datos

        user_data = get_user_input()

        # Transform the data into a dataframe
        def get_features():
            features = pd.DataFrame(user_data, index=[0])
            return features
        with st.container():    
            st.write("### Parámetros ingresados")
        col3, col4 = st.columns([2, 1])

        with col3:
            # Mostrar los valores ingresados
            st.write(f"Edad: {user_data['age']}")
            st.write(f"Sexo: {'Hombre' if user_data['gender'] == 1 else 'Mujer'}")
            st.write(f"Dolor en el pecho: {user_data['chestpain']}")
            st.write(f"Presión arterial en reposo: {user_data['restingBP']} mmHg")
            st.write(f"Colesterol: {user_data['serumcholestrol']} mm/dl")
            st.write(f"Azúcar en sangre > 120 mg/dl: {'Sí' if user_data['fastingbloodsugar'] == True else 'No'}")
            st.write(f"Electrocardiograma en reposo: {user_data['restingrelectro']}")
            st.write(f"Frecuencia cardiaca máxima: {user_data['maxheartrate']}")
            st.write(f"Angina producida por el ejercicio: {'Sí' if user_data['exerciseangia'] == 1 else 'No'}")
            st.write(f"Oldpeak: {user_data['oldpeak']}")
            st.write(f"Pendiente del segmento ST: {user_data['slope']}")
            st.write(f"Número de vasos obstruidos: {user_data['noofmajorvessels']}")

        user_input = get_features()

        with col4:
            st.write("Si está conforme con los datos presione el botón Evaluar")

            # Botón "Evaluar" en la parte inferior

            if st.button("Evaluar"):
                prediction = model.predict(user_input)
                probability = model.predict_proba(user_input)
                argmax = np.argmax(probability)
                probability = probability[0]

                # st.subheader('Input Data')
                # st.write(user_input)
                st.subheader('Resultado')
                classification_result = str(prediction)
                if argmax == 0:
                    classification_result = "Ningún riesgo encontrado"
                else:
                    classification_result = "Posible riesgo encontrado"

                st.success(classification_result)
                st.subheader('Accuracy')
                st.success(str((probability[argmax] * 100).round(2)) + "%")
    with tab2:
        st.header('Evaluar datos cargados de un diagnóstico')
        uploaded_file = st.file_uploader(
        "Subir los datos:", type=["csv"]
        )

        if uploaded_file:
            st.subheader('Resultados')
            df = pd.read_csv(uploaded_file, float_precision="round_trip")

            X = df.iloc[:, 1:13].values
            prediction = model.predict(X)
            probability = model.predict_proba(X)
            argmax = np.argmax(probability)
            # probability = probability[0]

            df2 = df[["age",
                    "gender",
                    "chestpain",
                    "restingBP",
                    "serumcholestrol",
                    "fastingbloodsugar",
                    "restingrelectro",
                    "maxheartrate",
                    "exerciseangia",
                    "oldpeak",
                    "slope",
                    "noofmajorvessels"]]

            pred = []
            for i in prediction:
                if i == 0:
                    pred.append("No heart disease")
                else:
                    pred.append("Heart disease risk")

            no_diabetic_accuracy = []
            diabetic_accuracy = []

            for i in probability[:, 0]:
                no_diabetic_accuracy.append(str((i * 100).round(2)) + "%")

            for i in probability[:, 1]:
                diabetic_accuracy.append(str((i * 100).round(2)) + "%")

            df2['Result'] = pred
            df2['No heart disease accuracy'] = no_diabetic_accuracy
            df2['Heart disease accuracy'] = diabetic_accuracy
            st.write(df2)
# Cambiar el color de fondo de toda la aplicación
#Color principal fffcea 
background = """
            <style>
                .stApp {
                    background-color: #fa7676;
                    opacity: 0.7;
                    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #fa7676 18px ),
                    repeating-linear-gradient( #ff003255, #ff0032 );
                }
                
            </style>
            """
column_design = """
            <style>
                .stMainBlockContainer {
                    background-color: #ffffff; 
                    border-radius: 10px;
                }
            </style>
            """
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            #header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown(background, unsafe_allow_html=True)
st.markdown(column_design, unsafe_allow_html=True)