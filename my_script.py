import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# Einführung auf der Startseite
st.title("Straßen-Scanner: Einblicke in die Straßenverhältnisse mit Beschleunigungsmetriken")
st.write("""
Willkommen zur Straßenbelagsanalyse. Hier können verschiedene Datensätze analysiert und Machine-Learning-Modelle getestet werden, um Straßenoberflächen zu klassifizieren. Bitte wählen Sie eine Option aus dem Menü, um fortzufahren.
""")

# Bild unter dem Text hinzufügen
image_path = "/Users/olivialawinski/IKT/analytics/Semesterabgabe/ikt-semesterabgabe/RoadScanner.webp" 
st.image(image_path, caption="Straßenanalyse", use_column_width=True)

# Sidebar für Auswahl der verschiedenen Seiten/Optionen
st.sidebar.title("Navigation")
option = st.sidebar.selectbox("Wählen Sie eine Option:", 
                              ["Einführung","Datensatz","Datenanalyse", "Modelltest", "Visualisierung", "Map"])

# Globale Datenlade-Funktion
def load_data():
    data_path = "/Users/olivialawinski/IKT/analytics/Semesterabgabe/ikt-semesterabgabe/combined_dataset_sorted.csv"
    try:
        data = pd.read_csv(data_path, delimiter=';', header=None)
        data.columns = ['EPOCH_TIME', 'DEVICE_NAME', 'SENSOR_TYPE', 'MEASUREMENT_VALUE', 'GPS_LAT', 'GPS_LONG', 'SOURCE_FOLDER']
        return data
    except FileNotFoundError:
        st.error(f"Die Datei unter {data_path} konnte nicht gefunden werden.")
        return None

# Funktion für das Piechart
def plot_pie_chart(data, title):
    surface_counts = data['surface_type'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(surface_counts, labels=surface_counts.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  
    plt.title(title)
    st.pyplot(fig)

# Inhalte je nach Auswahl anzeigen
if option == "Einführung":
    st.header("Einführung in das Projekt")
    st.write("""
    In diesem Projekt werden Straßenoberflächen mithilfe von Beschleunigungsdaten analysiert, die über ein Smartphone erfasst wurden. Das Ziel ist es, verschiedene Straßenbelagsqualitäten wie Asphalt, Pflastersteine oder beschädigte Straßen zu identifizieren.
    """)

elif option == "Datensatz":
    st.header("Datensatz")
    st.write("Hier können Sie Ihren eigenen Datensatz hochladen.")

    # Dateiupload-Funktion
    uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei", type="csv")

    # Überprüfen, ob eine Datei hochgeladen wurde
    if uploaded_file is not None:
        try:
            # CSV-Datei einlesen
            user_data = pd.read_csv(uploaded_file)
            
            
            st.write("Erfolgreich hochgeladene Datei:")
            st.write(user_data.head())  

            
            st.write("Spalten des Datensatzes:")
            st.write(user_data.columns)
            
        except Exception as e:
            st.error(f"Fehler beim Einlesen der Datei: {e}")
    else:
        st.write("Bitte laden Sie eine CSV-Datei hoch, um fortzufahren.")
    
elif option == "Datenanalyse":
    st.header("Datenanalyse")
    st.write("Hier können Sie den Datensatz analysieren und visualisieren.")
    
    # Daten laden (nur für diese Option)
    data = load_data()
    if data is not None:
        # Filtere relevante Spalten (SENSOR_TYPE = 20 für Beschleunigungsdaten)
        acceleration_data = data[data['SENSOR_TYPE'] == '20']

        # Extrahiere die X, Y, Z-Werte aus 'MEASUREMENT_VALUE'
        acceleration_values = acceleration_data['MEASUREMENT_VALUE'].str.split(',', expand=True).astype(float)
        acceleration_values.columns = ['acceleration_x', 'acceleration_y', 'acceleration_z']

        # Entferne Ausreißer (z-Score)
        z_scores = stats.zscore(acceleration_values)
        filtered_data = acceleration_values[(abs(z_scores) < 3).all(axis=1)]

        # Entferne fehlende Werte mit .loc[]
        filtered_data = filtered_data.loc[filtered_data.notnull().all(axis=1)]

        # Grundlegende Statistiken
        st.subheader("Grundlegende Statistiken")
        selected_column = st.selectbox("Wähle eine Spalte für die Statistik:", ['acceleration_x', 'acceleration_y', 'acceleration_z'])
        st.write(filtered_data[selected_column].describe())

        # Histogramm
        st.subheader("Histogramm der Beschleunigungswerte")
        selected_column_hist = st.selectbox("Wähle eine Spalte für das Histogramm:", ['acceleration_x', 'acceleration_y', 'acceleration_z'])
        fig, ax = plt.subplots()
        ax.hist(filtered_data[selected_column_hist], bins=20, color='blue', edgecolor='black')
        st.pyplot(fig)

        # Streudiagramm
        st.subheader("Streudiagramm der Beschleunigungsdaten")
        x_axis = st.selectbox("Wähle die X-Achse:", ['acceleration_x', 'acceleration_y', 'acceleration_z'])
        y_axis = st.selectbox("Wähle die Y-Achse:", ['acceleration_x', 'acceleration_y', 'acceleration_z'], index=1)
        fig, ax = plt.subplots()
        ax.scatter(filtered_data[x_axis], filtered_data[y_axis], alpha=0.5)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        st.pyplot(fig)

        # Korrelationsmatrix
        st.subheader("Korrelationsmatrix der Beschleunigungsdaten")
        correlation_matrix = filtered_data[['acceleration_x', 'acceleration_y', 'acceleration_z']].corr()
        st.write(correlation_matrix)
        
        fig, ax = plt.subplots()
        cax = ax.matshow(correlation_matrix, cmap='coolwarm')
        fig.colorbar(cax)
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.columns)))
        ax.set_xticklabels(correlation_matrix.columns)
        ax.set_yticklabels(correlation_matrix.columns)
        plt.title("Korrelationsmatrix", pad=20)
        st.pyplot(fig)

elif option == "Modelltest":
    st.header("Modelltest")
    st.write("Testen Sie hier verschiedene Modelle zur Klassifikation der Straßenoberflächen.")
    
    # Daten laden 
    data = load_data()
    if data is not None:
        # relevante Spalten (SENSOR_TYPE = 20 für Beschleunigungsdaten) filtern
        acceleration_data = data[data['SENSOR_TYPE'] == '20']

        # X, Y, Z-Werte aus 'MEASUREMENT_VALUE'extrahieren
        acceleration_values = acceleration_data['MEASUREMENT_VALUE'].str.split(',', expand=True).astype(float)
        acceleration_values.columns = ['acceleration_x', 'acceleration_y', 'acceleration_z']

        # Ausreißer entfernen (z-Score)
        z_scores = stats.zscore(acceleration_values)
        filtered_data = acceleration_values[(abs(z_scores) < 3).all(axis=1)]

        # Modellwahl durch den Benutzer
        model_choice = st.selectbox("Wählen Sie ein Modell zur Klassifikation:", 
                                    ["k-nearest neighbors", "Random Forest"])

        # Aufteilen der Daten in Trainings- und Testdaten
        X = filtered_data[['acceleration_x', 'acceleration_y', 'acceleration_z']]
        y = np.random.randint(0, 2, X.shape[0])  # Beispielhafte Zielvariable (ersetzen mit tatsächlichen Zielwerten)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Auswahl und Konfiguration des Modells
        if model_choice == "k-nearest neighbors":
            # Auswahl der Hyperparameter für kNN
            n_neighbors = st.slider("Wählen Sie die Anzahl der Nachbarn für kNN", 1, 20, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        
        elif model_choice == "Random Forest":
            # Auswahl der Hyperparameter für Random Forest
            n_estimators = st.slider("Wählen Sie die Anzahl der Bäume für den Random Forest", 10, 100, 50)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        
        # Modell trainieren
        model.fit(X_train, y_train)
        
        # Vorhersagen und Genauigkeit
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f"Modell: {model_choice}")
        st.write(f"Modellgenauigkeit: {accuracy:.2f}")


elif option == "Visualisierung":
    st.header("Datenvisualisierung")
    
    # Daten laden
    data = load_data()
    if data is not None:
        # Filtere relevante Spalten (SENSOR_TYPE = 20 für Beschleunigungsdaten)
        acceleration_data = data[data['SENSOR_TYPE'] == '20']

        # Extrahiere die X, Y, Z-Werte aus 'MEASUREMENT_VALUE'
        acceleration_values = acceleration_data['MEASUREMENT_VALUE'].str.split(',', expand=True).astype(float)
        acceleration_values.columns = ['acceleration_x', 'acceleration_y', 'acceleration_z']

        # Füge eine Spalte für "surface_type" hinzu (Beispielhafte Klassifizierung, anpassen)
        surface_types = np.random.choice(['Asphalt', 'Pflastersteine', 'Uneben'], size=acceleration_values.shape[0])
        acceleration_values['surface_type'] = surface_types
        
        # Zeige Piechart für alle Daten (vor Filterung)
        st.subheader("Straßentypen - Vor Filterung")
        plot_pie_chart(acceleration_values, "Straßentypen vor Filterung")
        
        # Entferne Ausreißer und filtere nach minimaler Beschleunigung
        z_scores = stats.zscore(acceleration_values[['acceleration_x', 'acceleration_y', 'acceleration_z']])
        filtered_data = acceleration_values[(abs(z_scores) < 3).all(axis=1)]
        min_acc = st.slider(
            "Minimale Beschleunigung (X-Achse)", 
            float(filtered_data['acceleration_x'].min()), 
            float(filtered_data['acceleration_x'].max())
        )
        filtered_data = filtered_data[filtered_data['acceleration_x'] >= min_acc]

        # Zeige Piechart für gefilterte Daten
        st.subheader("Straßentypen - Nach Filterung")
        plot_pie_chart(filtered_data, "Straßentypen nach Filterung")

elif option == "Map":
    st.header("Map")
    st.write("Hier können Sie anhand der GPS-Daten sehen, wo die Messungen stattgefunden haben.") 

    # Daten laden (nur für diese Option)
    data = load_data()
    if data is not None:
        # Konvertiere GPS_LAT und GPS_LONG in numerische Werte (float)
        data['GPS_LAT'] = pd.to_numeric(data['GPS_LAT'], errors='coerce')
        data['GPS_LONG'] = pd.to_numeric(data['GPS_LONG'], errors='coerce')

        # Entferne Zeilen mit nicht-numerischen GPS-Werten (NaN)
        data = data.dropna(subset=['GPS_LAT', 'GPS_LONG'])

        # Filter-Optionen innerhalb eines Formulars
        with st.form("map_filter_form"):
            st.subheader("Filtere nach GPS-Koordinaten")
            
            # Definiere initiale Werte für den Filter basierend auf den tatsächlichen GPS-Daten
            min_lat = st.number_input("Minimale Breite", value=float(data['GPS_LAT'].min()))
            max_lat = st.number_input("Maximale Breite", value=float(data['GPS_LAT'].max()))
            min_long = st.number_input("Minimale Länge", value=float(data['GPS_LONG'].min()))
            max_long = st.number_input("Maximale Länge", value=float(data['GPS_LONG'].max()))
            
            submitted = st.form_submit_button("Filter anwenden")

        if submitted:
            # Filtere die GPS-Daten basierend auf den benutzerspezifischen Werten
            filtered_gps_data = data[(data['GPS_LAT'] >= min_lat) & (data['GPS_LAT'] <= max_lat) & 
                                     (data['GPS_LONG'] >= min_long) & (data['GPS_LONG'] <= max_long)]
    
            # Zeige die Anzahl der gefilterten Daten an
            st.write(f"Gefilterte GPS-Daten: {len(filtered_gps_data)} Einträge")
            
            # Überprüfe, ob gefilterte Daten vorhanden sind
            if len(filtered_gps_data) > 0:
                # Optional: Downsampling für große Datensätze zur Verbesserung der Performance
                max_points = 1000
                if len(filtered_gps_data) > max_points:
                    filtered_gps_data = filtered_gps_data.sample(n=max_points, random_state=42)
                    st.write(f"Die Daten wurden auf {max_points} Einträge heruntergesampelt zur besseren Performance.")
                
                # Umbenennen der Spalten in die erwarteten Namen für st.map()
                filtered_gps_data = filtered_gps_data.rename(columns={'GPS_LAT': 'latitude', 'GPS_LONG': 'longitude'})
                
                # Map Function - Zeige die gefilterten GPS-Daten auf der Karte
                st.map(filtered_gps_data[['latitude', 'longitude']])
            else:
                st.write("Keine GPS-Daten in diesem Bereich gefunden.")
    else:
        st.write("Keine Daten verfügbar.")







        






    









