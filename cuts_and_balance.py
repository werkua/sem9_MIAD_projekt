import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

track_files = [
    "Data/atm_neutrino_classB.h5",
    "Data/atm_neutrino_classF.h5"
]

shower_files = [
    "Data/atm_neutrino_classC.h5",
    "Data/atm_neutrino_classG.h5",
    "Data/atm_neutrino_classA.h5",
    "Data/atm_neutrino_classE.h5"
]

output_dir = "ML_Ready_Data"
plot_dir = "plots_cleaned"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# --- DEFINICJA CIĘĆ ---

def apply_cuts(df):

    # Zmienna time - usuniety pre-trigger (wartosci < 0) i długie ogony
    df = df[ (df['time'] > 0) & (df['time'] < 2500) ]
    
    # Zmienna ToT - usuniete wartości < 10, ktore najprawdopodobnie sa szumem (dark noise)
    df = df[ df['tot'] > 10 ]
    
    # Zmienne położenia - usuniete zdazenia z brzegow detektora (szumy, promieniowanie z materialow
    # konstrukcji detektora, bledna rekonstrukcja zdarzen brzegowych???)
    # Można te warunki usunąć i sprawdzić po ML jak to siada
    df = df[ (df['pos_x'] > -90) & (df['pos_x'] < 90) ]
    df = df[ (df['pos_y'] > -110) & (df['pos_y'] < 110) ]
    df = df[ (df['pos_z'] > 40) & (df['pos_z'] < 190) ]
    
    return df

# ---- BALANS GRUP -----

def load_and_process_group(file_list, group_name):
    dataframes = []
    print(f"\n--- Wczytywanie grupy: {group_name} ---")
    for f in file_list:
        try:
            df = pd.read_hdf(f)
            dataframes.append(df)
            print(f"Załadowano {os.path.basename(f)}: {len(df)} zdarzeń")
        except Exception as e:
            print(f"Błąd przy pliku {f}: {e}")
    
    if not dataframes:
        print(f"Ostrzeżenie: Brak danych dla grupy {group_name}!")
        return pd.DataFrame()

    full_df = pd.concat(dataframes, ignore_index=True)
    print(f"Razem przed cięciami: {len(full_df)}")
    
    clean_df = apply_cuts(full_df)
    print(f"Razem po cięciach: {len(clean_df)} (odrzucono {len(full_df) - len(clean_df)})")
    
    return clean_df

# ----- WYKONANIE BALANSU I CIĘĆ -------

df_track = load_and_process_group(track_files, "TRACK")
df_shower = load_and_process_group(shower_files, "SHOWER")

# Balansowanie danych
n_track = len(df_track)
n_shower = len(df_shower)
min_samples = min(n_track, n_shower)

print(f"\n--- Balansowanie klas ---")
print(f"Liczba Track: {n_track}, Liczba Shower: {n_shower}")
print(f"Redukuję obie klasy do liczby: {min_samples}")

# Losujemy próbkę o liczebności mniejszej klasy
# random_state zapewnia powtarzalność wyników
df_track_balanced = df_track.sample(n=min_samples, random_state=42)
df_shower_balanced = df_shower.sample(n=min_samples, random_state=42)

# Etykiety
df_track_balanced['class_label'] = 0  # 0 dla Track
df_shower_balanced['class_label'] = 1 # 1 dla Shower

# Zapis do CSV
path_track = os.path.join(output_dir, "dataset_track_balanced.csv")
path_shower = os.path.join(output_dir, "dataset_shower_balanced.csv")

df_track_balanced.to_csv(path_track, index=False)
df_shower_balanced.to_csv(path_shower, index=False)

print(f"\nZapisano pliki przygotowane do ML:")
print(f" -> {path_track}")
print(f" -> {path_shower}")

# --- PLOTOWANIE PO CIECIACH ----

print("\nGenerowanie wykresów porównawczych...")
variables_to_plot = ["pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z", "time", "tot"]

for var in variables_to_plot:
    plt.figure(figsize=(8, 6))

    plt.hist(df_track_balanced[var], bins=50, alpha=0.5, label='Track', density=False, color='blue')
    plt.hist(df_shower_balanced[var], bins=50, alpha=0.5, label='Shower', density=False, color='orange')
    
    plt.title(f"Rozkład zmiennej: {var} (po cięciach i balansowaniu)")
    plt.xlabel(var)
    plt.ylabel("Liczba zdarzeń")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale('log') 
    
    save_path = os.path.join(plot_dir, f"compare_{var}.png")
    plt.savefig(save_path)
    plt.close()

print(f"Wykresy zapisano w katalogu: {plot_dir}")