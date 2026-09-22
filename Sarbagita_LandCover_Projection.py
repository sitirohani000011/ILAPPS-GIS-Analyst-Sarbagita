#!/usr/bin/env python
# coding: utf-8

# # Sarbagita Land-Cover Classification and Projection
# 
# ## GIS Analyst Take-Home Assignment
# 
# ### Study Area
# 
# Sarbagita Agglomeration, Bali, Indonesia:
# - Denpasar
# - Badung
# - Gianyar
# - Tabanan
# 
# ### Objective
# 
# This project develops a Python-based workflow for land-cover classification
# and projection using Sentinel-derived raster data.
# 
# The workflow includes:
# 
# 1. Land-cover classification for 2020 and 2023.
# 2. Accuracy assessment.
# 3. Land-cover transition analysis.
# 4. Markov-based projection for 2026.
# 5. Spatial allocation of projected land-cover changes.
# 6. Land-cover area and change analysis.
# 7. Export of final results.
# 
# ### Land-Cover Classes
# 
# | Classvalue | Land-Cover Class |
# |---:|---|
# | 1 | Pemukiman atau Lahan Terbangun |
# | 21 | Lahan Kosong |
# | 35 | Tubuh Air |
# | 40 | Pertanian |
# | 56 | Vegetasi |
# 
# ### Data Limitation
# 
# The supplied raster datasets contain three visible bands (RGB).
# Therefore, RGB predictors are used for classification. NIR-dependent
# indices such as NDVI are not calculated because an NIR band is not
# available in the supplied raster data.

# In[1]:


import os
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.features import geometry_mask

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix
)

from scipy.ndimage import uniform_filter
from pyproj import Geod

from IPython.display import display

print("Libraries successfully imported.")


# In[2]:


from pathlib import Path

# Folder utama proyek
BASE_DIR = Path("/mnt/d/ILAPS")

# Folder data input
INPUT_DIR = (
    BASE_DIR
    / "Data Citra Sentinel Sarbagita-20260921T123102Z-1-001"
    / "Data Citra Sentinel Sarbagita"
)

# Folder training sample
TRAINING_DIR = (
    INPUT_DIR
    / "Training Sample"
    / "Training Sample"
)

# Folder untuk hasil akhir
OUTPUT_DIR = BASE_DIR / "outputs"

# Buat folder output jika belum ada
OUTPUT_DIR.mkdir(exist_ok=True)

# File input
RASTER_2020 = INPUT_DIR / "Sarbagita_2020.tif"
RASTER_2023 = INPUT_DIR / "Sarbagita_2023.tif"

TRAINING_SHP = TRAINING_DIR / "Training Sample_tif.shp"

print("BASE_DIR     :", BASE_DIR)
print("INPUT_DIR    :", INPUT_DIR)
print("OUTPUT_DIR   :", OUTPUT_DIR)
print()
print("Raster 2020  :", RASTER_2020)
print("Raster 2023  :", RASTER_2023)
print("Training SHP :", TRAINING_SHP)


# In[3]:


# ============================================================
# CHECK INPUT FILES
# ============================================================

files_to_check = {
    "Sarbagita 2020": RASTER_2020,
    "Sarbagita 2023": RASTER_2023,
    "Training sample": TRAINING_SHP,
}

for name, path in files_to_check.items():
    status = "OK" if path.exists() else "NOT FOUND"
    print(f"{name:20s}: {status}")


# In[4]:


# ============================================================
# LOAD TRAINING SAMPLE
# ============================================================

training = gpd.read_file(TRAINING_SHP)

print("Jumlah training sample :", len(training))
print("CRS                    :", training.crs)
print("Kolom                  :", list(training.columns))

display(training)


# In[5]:


# ============================================================
# CHECK LAND-COVER CLASSES
# ============================================================

class_table = (
    training[["Classvalue", "Classname"]]
    .drop_duplicates()
    .sort_values("Classvalue")
    .reset_index(drop=True)
)

display(class_table)


# In[6]:


# ============================================================
# LOAD AND CHECK RASTER DATA
# ============================================================

with rasterio.open(RASTER_2020) as src:
    raster_2020 = src.read()
    profile_2020 = src.profile.copy()
    
    print("=== RASTER 2020 ===")
    print("Shape      :", raster_2020.shape)
    print("CRS        :", src.crs)
    print("Resolution :", src.res)
    print("Dtype      :", src.dtypes)
    print("NoData     :", src.nodata)
    print("Bounds     :", src.bounds)
    print("Bands      :", src.count)
    print("Descriptions:", src.descriptions)


with rasterio.open(RASTER_2023) as src:
    raster_2023 = src.read()
    profile_2023 = src.profile.copy()
    
    print("\n=== RASTER 2023 ===")
    print("Shape      :", raster_2023.shape)
    print("CRS        :", src.crs)
    print("Resolution :", src.res)
    print("Dtype      :", src.dtypes)
    print("NoData     :", src.nodata)
    print("Bounds     :", src.bounds)
    print("Bands      :", src.count)
    print("Descriptions:", src.descriptions)


# In[7]:


# ============================================================
# EXTRACT TRAINING SAMPLES FROM 2020 RASTER
# ============================================================

with rasterio.open(RASTER_2020) as src:

    # Baca RGB raster
    image_2020 = src.read()

    # Pastikan CRS training sama dengan raster
    training_2020 = training.to_crs(src.crs)

    X_list = []
    y_list = []

    for _, row in training_2020.iterrows():

        # Mask area polygon training sample
        mask = geometry_mask(
            [row.geometry],
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width)
        )

        # Ambil pixel RGB di dalam polygon
        pixels = image_2020[:, mask].T

        # Buang pixel yang tidak valid jika ada
        pixels = pixels[np.all(np.isfinite(pixels), axis=1)]

        # Simpan predictor RGB
        X_list.append(pixels)

        # Simpan label kelas
        y_list.append(
            np.full(
                pixels.shape[0],
                row["Classvalue"],
                dtype=np.int16
            )
        )

# Gabungkan seluruh training sample
X = np.vstack(X_list)
y = np.concatenate(y_list)

print("Jumlah training pixels :", len(y))
print("Jumlah predictor       :", X.shape[1])

print("\nDistribusi kelas:")
print(
    pd.Series(y)
    .value_counts()
    .sort_index()
)

print("\nContoh predictor RGB:")
print(X[:5])


# In[8]:


# ============================================================
# SPLIT TRAINING AND VALIDATION DATA
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples   :", len(y_train))
print("Validation samples :", len(y_val))
print("Total samples      :", len(y))


# In[9]:


# ============================================================
# TRAIN RANDOM FOREST MODEL - 2020
# ============================================================

rf_2020 = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf_2020.fit(X_train, y_train)

print("Random Forest 2020 berhasil dilatih.")
print("Jumlah trees :", rf_2020.n_estimators)


# In[10]:


# ============================================================
# VALIDATE RANDOM FOREST MODEL - 2020
# ============================================================

y_pred_2020 = rf_2020.predict(X_val)

# Overall Accuracy
oa_2020 = accuracy_score(y_val, y_pred_2020)

# Cohen's Kappa
kappa_2020 = cohen_kappa_score(y_val, y_pred_2020)

print("=== ACCURACY ASSESSMENT 2020 ===")
print(f"Overall Accuracy : {oa_2020:.4f}")
print(f"Cohen's Kappa    : {kappa_2020:.4f}")


# In[11]:


# ============================================================
# CLASS-LEVEL ACCURACY ASSESSMENT - 2020
# ============================================================

classes = sorted(np.unique(y_val))

cm_2020 = confusion_matrix(
    y_val,
    y_pred_2020,
    labels=classes
)

# Producer's Accuracy = diagonal / total actual
producer_accuracy_2020 = np.diag(cm_2020) / cm_2020.sum(axis=1)

# User's Accuracy = diagonal / total predicted
user_accuracy_2020 = np.diag(cm_2020) / cm_2020.sum(axis=0)

accuracy_2020 = pd.DataFrame({
    "Classvalue": classes,
    "Classname": [
        class_table.loc[
            class_table["Classvalue"] == c, "Classname"
        ].iloc[0]
        for c in classes
    ],
    "Producer_Accuracy": producer_accuracy_2020,
    "User_Accuracy": user_accuracy_2020
})

print("=== CONFUSION MATRIX 2020 ===")
print(cm_2020)

print("\n=== CLASS-LEVEL ACCURACY 2020 ===")
display(accuracy_2020)


# In[12]:


# ============================================================
# FULL RASTER CLASSIFICATION - 2020
# RAM-SAFE CHUNKED PREDICTION
# ============================================================

OUTPUT_2020 = OUTPUT_DIR / "Sarbagita_LandCover_2020.tif"

with rasterio.open(RASTER_2020) as src:

    profile = src.profile.copy()

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(OUTPUT_2020, "w", **profile) as dst:

        total_pixels = src.width * src.height
        processed_pixels = 0

        # Proses raster per block
        for _, window in src.block_windows(1):

            # Baca RGB hanya untuk satu block
            image_block = src.read(window=window)

            # Ubah menjadi format:
            # (jumlah pixel, jumlah predictor)
            X_block = image_block.reshape(
                image_block.shape[0],
                -1
            ).T

            # Prediksi block
            y_block = rf_2020.predict(X_block)

            # Kembalikan ke bentuk raster 2D
            classified_block = y_block.reshape(
                image_block.shape[1],
                image_block.shape[2]
            )

            # Simpan hasil block
            dst.write(
                classified_block.astype(np.uint8),
                1,
                window=window
            )

            processed_pixels += X_block.shape[0]

            progress = processed_pixels / total_pixels * 100

            print(
                f"Progress: {progress:6.2f}%",
                end="\r"
            )

print()
print("Klasifikasi 2020 selesai.")
print("Output:", OUTPUT_2020)


# In[13]:


# ============================================================
# CHECK CLASSIFIED LAND-COVER - 2020
# ============================================================

with rasterio.open(OUTPUT_2020) as src:
    classified_check = src.read(1)

print("=== OUTPUT 2020 ===")
print("File       :", OUTPUT_2020)
print("Shape      :", classified_check.shape)
print("CRS        :", src.crs)
print("Data type :", src.dtypes[0])

# Hitung jumlah pixel setiap kelas
class_counts_2020 = (
    pd.Series(classified_check.ravel())
    .value_counts()
    .sort_index()
)

print("\n=== DISTRIBUSI KELAS 2020 ===")
display(
    class_counts_2020.rename("Pixel Count").to_frame()
)


# In[14]:


# ============================================================
# EXTRACT TRAINING SAMPLES FROM 2023 RASTER
# ============================================================

with rasterio.open(RASTER_2023) as src:

    image_2023 = src.read()
    training_2023 = training.to_crs(src.crs)

    X_list_2023 = []
    y_list_2023 = []

    for _, row in training_2023.iterrows():

        # Mask polygon training sample
        mask = geometry_mask(
            [row.geometry],
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width)
        )

        # Ambil pixel RGB
        pixels = image_2023[:, mask].T

        # Buang pixel yang tidak valid
        pixels = pixels[np.all(np.isfinite(pixels), axis=1)]

        X_list_2023.append(pixels)

        # Label kelas
        y_list_2023.append(
            np.full(
                pixels.shape[0],
                row["Classvalue"],
                dtype=np.int16
            )
        )

# Gabungkan seluruh sample
X_2023 = np.vstack(X_list_2023)
y_2023 = np.concatenate(y_list_2023)

print("Jumlah training pixels :", len(y_2023))
print("Jumlah predictor       :", X_2023.shape[1])

print("\nDistribusi kelas 2023:")
print(
    pd.Series(y_2023)
    .value_counts()
    .sort_index()
)


# In[15]:


# ============================================================
# SPLIT TRAINING AND VALIDATION DATA - 2023
# ============================================================

X_train_2023, X_val_2023, y_train_2023, y_val_2023 = train_test_split(
    X_2023,
    y_2023,
    test_size=0.20,
    random_state=42,
    stratify=y_2023
)

print("Training samples   :", len(y_train_2023))
print("Validation samples :", len(y_val_2023))
print("Total samples      :", len(y_2023))


# In[16]:


# ============================================================
# TRAIN RANDOM FOREST MODEL - 2023
# ============================================================

rf_2023 = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf_2023.fit(X_train_2023, y_train_2023)

print("Random Forest 2023 berhasil dilatih.")
print("Jumlah trees :", rf_2023.n_estimators)


# In[17]:


# ============================================================
# VALIDATE RANDOM FOREST MODEL - 2023
# ============================================================

y_pred_2023 = rf_2023.predict(X_val_2023)

# Overall Accuracy
oa_2023 = accuracy_score(y_val_2023, y_pred_2023)

# Cohen's Kappa
kappa_2023 = cohen_kappa_score(y_val_2023, y_pred_2023)

print("=== ACCURACY ASSESSMENT 2023 ===")
print(f"Overall Accuracy : {oa_2023:.4f}")
print(f"Cohen's Kappa    : {kappa_2023:.4f}")


# In[18]:


# ============================================================
# CLASS-LEVEL ACCURACY ASSESSMENT - 2023
# ============================================================

classes_2023 = sorted(np.unique(y_val_2023))

cm_2023 = confusion_matrix(
    y_val_2023,
    y_pred_2023,
    labels=classes_2023
)

# Producer's Accuracy
producer_accuracy_2023 = (
    np.diag(cm_2023) / cm_2023.sum(axis=1)
)

# User's Accuracy
user_accuracy_2023 = (
    np.diag(cm_2023) / cm_2023.sum(axis=0)
)

accuracy_2023 = pd.DataFrame({
    "Classvalue": classes_2023,
    "Classname": [
        class_table.loc[
            class_table["Classvalue"] == c, "Classname"
        ].iloc[0]
        for c in classes_2023
    ],
    "Producer_Accuracy": producer_accuracy_2023,
    "User_Accuracy": user_accuracy_2023
})

print("=== CONFUSION MATRIX 2023 ===")
print(cm_2023)

print("\n=== CLASS-LEVEL ACCURACY 2023 ===")
display(accuracy_2023)


# In[19]:


# ============================================================
# FULL RASTER CLASSIFICATION - 2023
# RAM-SAFE CHUNKED PREDICTION
# ============================================================

OUTPUT_2023 = OUTPUT_DIR / "Sarbagita_LandCover_2023.tif"

with rasterio.open(RASTER_2023) as src:

    profile = src.profile.copy()

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(OUTPUT_2023, "w", **profile) as dst:

        total_pixels = src.width * src.height
        processed_pixels = 0

        # Proses raster per block
        for _, window in src.block_windows(1):

            # Baca RGB satu block
            image_block = src.read(window=window)

            # Ubah menjadi:
            # (jumlah pixel, jumlah predictor)
            X_block = image_block.reshape(
                image_block.shape[0],
                -1
            ).T

            # Prediksi block
            y_block = rf_2023.predict(X_block)

            # Kembalikan menjadi raster 2D
            classified_block = y_block.reshape(
                image_block.shape[1],
                image_block.shape[2]
            )

            # Simpan hasil
            dst.write(
                classified_block.astype(np.uint8),
                1,
                window=window
            )

            processed_pixels += X_block.shape[0]

            progress = processed_pixels / total_pixels * 100

            print(
                f"Progress: {progress:6.2f}%",
                end="\r"
            )

print()
print("Klasifikasi 2023 selesai.")
print("Output:", OUTPUT_2023)


# In[20]:


# ============================================================
# CHECK CLASSIFIED LAND-COVER - 2023
# ============================================================

with rasterio.open(OUTPUT_2023) as src:
    classified_check_2023 = src.read(1)

print("=== OUTPUT 2023 ===")
print("File       :", OUTPUT_2023)
print("Shape      :", classified_check_2023.shape)
print("CRS        :", src.crs)
print("Data type :", src.dtypes[0])

# Hitung jumlah pixel setiap kelas
class_counts_2023 = (
    pd.Series(classified_check_2023.ravel())
    .value_counts()
    .sort_index()
)

print("\n=== DISTRIBUSI KELAS 2023 ===")
display(
    class_counts_2023.rename("Pixel Count").to_frame()
)


# In[21]:


# ============================================================
# LAND-COVER TRANSITION MATRIX
# 2020 -> 2023
# ============================================================

with rasterio.open(OUTPUT_2020) as src:
    lc_2020 = src.read(1)

with rasterio.open(OUTPUT_2023) as src:
    lc_2023 = src.read(1)

# Kelas land-cover
classes = [1, 21, 35, 40, 56]

# Buat transition matrix
transition_counts = pd.DataFrame(
    0,
    index=classes,
    columns=classes,
    dtype=np.int64
)

for from_class in classes:
    for to_class in classes:

        transition_counts.loc[from_class, to_class] = np.sum(
            (lc_2020 == from_class) &
            (lc_2023 == to_class)
        )

print("=== TRANSITION MATRIX (PIXEL COUNT) ===")
display(transition_counts)


# In[22]:


# ============================================================
# TRANSITION PROBABILITY MATRIX
# 2020 -> 2023
# ============================================================

transition_probability = (
    transition_counts.div(
        transition_counts.sum(axis=1),
        axis=0
    )
)

print("=== TRANSITION PROBABILITY MATRIX ===")
display(
    transition_probability.round(4)
)

print("\n=== CHECK ROW SUM ===")
print(
    transition_probability.sum(axis=1).round(4)
)


# In[24]:


# ============================================================
# MARKOV PROJECTION - 2026
# ============================================================

# Jumlah pixel setiap kelas pada tahun 2023
current_counts_2023 = (
    pd.Series(lc_2023.ravel())
    .value_counts()
    .reindex(classes)
    .fillna(0)
    .astype(np.int64)
)

# Proyeksi jumlah pixel 2026
projected_counts_2026 = (
    current_counts_2023.values
    @ transition_probability.values
)

# Bulatkan menjadi jumlah pixel integer
projected_counts_2026 = np.rint(
    projected_counts_2026
).astype(np.int64)

# Total pixel raster yang benar
total_pixels = lc_2023.size

# Koreksi pembulatan agar total pixel tetap sama
difference = (
    total_pixels -
    projected_counts_2026.sum()
)

if difference != 0:
    projected_counts_2026[np.argmax(projected_counts_2026)] += difference

markov_projection = pd.DataFrame({
    "Classvalue": classes,
    "Current_2023_Pixels": current_counts_2023.values,
    "Projected_2026_Pixels": projected_counts_2026
})

# Hitung perubahan
markov_projection["Change_Pixels"] = (
    markov_projection["Projected_2026_Pixels"]
    - markov_projection["Current_2023_Pixels"]
)

print("=== MARKOV PROJECTION 2026 ===")
display(markov_projection)

print("\nTotal pixels 2023 :", current_counts_2023.sum())
print("Total pixels 2026 :", projected_counts_2026.sum())
print("Raster pixel total:", total_pixels)


# In[26]:


# ============================================================
# LAND-COVER GAIN / LOSS ANALYSIS
# 2023 -> 2026
# ============================================================

gain_loss_2026 = markov_projection.copy()

# Tambahkan nama kelas
gain_loss_2026["Classname"] = [
    class_table.loc[
        class_table["Classvalue"] == c,
        "Classname"
    ].iloc[0]
    for c in gain_loss_2026["Classvalue"]
]

# Hitung perubahan persentase
gain_loss_2026["Change_Percent"] = (
    gain_loss_2026["Change_Pixels"]
    / gain_loss_2026["Current_2023_Pixels"]
    * 100
)

print("=== LAND-COVER GAIN / LOSS 2023 -> 2026 ===")

display(
    gain_loss_2026[
        [
            "Classvalue",
            "Classname",
            "Current_2023_Pixels",
            "Projected_2026_Pixels",
            "Change_Pixels",
            "Change_Percent"
        ]
    ].round(2)
)


# In[27]:


# ============================================================
# NEIGHBORHOOD SUITABILITY - 2023
# ============================================================

# Ukuran neighborhood
WINDOW_SIZE = 5

# Dictionary untuk menyimpan suitability setiap kelas
suitability = {}

for target_class in classes:

    # Mask kelas target pada tahun 2023
    target_mask = (
        lc_2023 == target_class
    ).astype(np.float32)

    # Hitung proporsi kelas target dalam neighborhood
    suitability[target_class] = uniform_filter(
        target_mask,
        size=WINDOW_SIZE,
        mode="nearest"
    )

print("=== NEIGHBORHOOD SUITABILITY ===")
print("Window size :", f"{WINDOW_SIZE} x {WINDOW_SIZE}")

for target_class in classes:
    print(
        f"Class {target_class}: "
        f"min={suitability[target_class].min():.4f}, "
        f"max={suitability[target_class].max():.4f}, "
        f"mean={suitability[target_class].mean():.4f}"
    )


# In[28]:


# ============================================================
# IDENTIFY GAIN AND LOSS CLASSES
# ============================================================

# Kelas yang bertambah
gain_classes = gain_loss_2026.loc[
    gain_loss_2026["Change_Pixels"] > 0,
    "Classvalue"
].tolist()

# Kelas yang berkurang
loss_classes = gain_loss_2026.loc[
    gain_loss_2026["Change_Pixels"] < 0,
    "Classvalue"
].tolist()

# Besarnya gain
gain_targets = {
    int(row["Classvalue"]): int(row["Change_Pixels"])
    for _, row in gain_loss_2026.iterrows()
    if row["Change_Pixels"] > 0
}

# Besarnya loss
loss_targets = {
    int(row["Classvalue"]): int(-row["Change_Pixels"])
    for _, row in gain_loss_2026.iterrows()
    if row["Change_Pixels"] < 0
}

print("=== GAIN CLASSES ===")
print(gain_targets)

print("\n=== LOSS CLASSES ===")
print(loss_targets)

print("\nTotal gain :", sum(gain_targets.values()))
print("Total loss :", sum(loss_targets.values()))


# In[30]:


# ============================================================
# ALLOCATION QUOTA: LOSS CLASSES -> GAIN CLASSES
# WITH EXACT SOURCE AND TARGET TOTALS
# ============================================================

# Matriks bobot berdasarkan transition probability
weight_matrix = pd.DataFrame(
    0.0,
    index=loss_classes,
    columns=gain_classes
)

for source_class in loss_classes:
    for target_class in gain_classes:

        weight_matrix.loc[
            source_class,
            target_class
        ] = transition_probability.loc[
            source_class,
            target_class
        ]

# ------------------------------------------------------------
# Karena hanya ada 2 source dan 3 target,
# kita gunakan proporsi transition probability
# dengan constraint total source dan target.
# ------------------------------------------------------------

allocation_quota = pd.DataFrame(
    0,
    index=loss_classes,
    columns=gain_classes,
    dtype=np.int64
)

# Target gain yang harus dipenuhi
target_remaining = gain_targets.copy()

# Source loss yang harus dipenuhi
source_remaining = loss_targets.copy()

# Urutkan pasangan berdasarkan transition probability
pairs = []

for source_class in loss_classes:
    for target_class in gain_classes:

        pairs.append(
            (
                transition_probability.loc[
                    source_class,
                    target_class
                ],
                source_class,
                target_class
            )
        )

# Prioritaskan transition probability terbesar
pairs.sort(reverse=True)

# Alokasikan secara greedy dengan tetap menjaga
# sisa kebutuhan source dan target
for probability, source_class, target_class in pairs:

    amount = min(
        source_remaining[source_class],
        target_remaining[target_class]
    )

    allocation_quota.loc[
        source_class,
        target_class
    ] = amount

    source_remaining[source_class] -= amount
    target_remaining[target_class] -= amount

# ------------------------------------------------------------
# Perbaikan residual menggunakan proporsi transition
# probability untuk source/target yang masih tersisa.
# ------------------------------------------------------------

while sum(source_remaining.values()) > 0:

    available_pairs = []

    for source_class in loss_classes:
        for target_class in gain_classes:

            if (
                source_remaining[source_class] > 0
                and target_remaining[target_class] > 0
            ):

                available_pairs.append(
                    (
                        transition_probability.loc[
                            source_class,
                            target_class
                        ],
                        source_class,
                        target_class
                    )
                )

    if not available_pairs:
        break

    available_pairs.sort(reverse=True)

    _, source_class, target_class = available_pairs[0]

    amount = min(
        source_remaining[source_class],
        target_remaining[target_class]
    )

    allocation_quota.loc[
        source_class,
        target_class
    ] += amount

    source_remaining[source_class] -= amount
    target_remaining[target_class] -= amount


print("=== FINAL ALLOCATION QUOTA ===")
display(allocation_quota)

print("\n=== SOURCE TOTAL ===")
display(
    allocation_quota.sum(axis=1).rename(
        "Allocated_Loss"
    ).to_frame()
)

print("\n=== TARGET TOTAL ===")
display(
    allocation_quota.sum(axis=0).rename(
        "Allocated_Gain"
    ).to_frame()
)

print("\nRemaining source:")
print(source_remaining)

print("\nRemaining target:")
print(target_remaining)


# In[31]:


# ============================================================
# VALIDATE ALLOCATION QUOTA
# ============================================================

source_totals = allocation_quota.sum(axis=1)
target_totals = allocation_quota.sum(axis=0)

print("=== SOURCE ALLOCATION CHECK ===")

for source_class in loss_classes:
    print(
        f"Class {source_class}: "
        f"allocated = {source_totals[source_class]:,} "
        f"| required = {loss_targets[source_class]:,}"
    )

print("\n=== TARGET ALLOCATION CHECK ===")

for target_class in gain_classes:
    print(
        f"Class {target_class}: "
        f"allocated = {target_totals[target_class]:,} "
        f"| required = {gain_targets[target_class]:,}"
    )

# Validasi otomatis
source_check = all(
    source_totals[c] == loss_targets[c]
    for c in loss_classes
)

target_check = all(
    target_totals[c] == gain_targets[c]
    for c in gain_classes
)

print("\nSource allocation valid :", source_check)
print("Target allocation valid :", target_check)

if source_check and target_check:
    print("\n✓ ALLOCATION QUOTA VALID")
else:
    print("\n✗ ALLOCATION QUOTA NEEDS ADJUSTMENT")


# In[32]:


# ============================================================
# PREPARE SPATIAL ALLOCATION CANDIDATES
# ============================================================

# Salin raster 2023 sebagai dasar proyeksi
lc_2026 = lc_2023.copy()

# Array untuk menandai pixel yang sudah dialokasikan
allocated_mask = np.zeros(
    lc_2023.shape,
    dtype=bool
)

# Simpan kandidat untuk setiap source-target pair
candidate_info = {}

for source_class in loss_classes:

    # Pixel yang berasal dari kelas loss
    source_mask = (
        lc_2023 == source_class
    )

    print(
        f"\nSource class {source_class}: "
        f"{source_mask.sum():,} pixels"
    )

    for target_class in gain_classes:

        quota = allocation_quota.loc[
            source_class,
            target_class
        ]

        if quota == 0:
            continue

        # Suitability target di lokasi source
        suitability_score = suitability[target_class]

        # Transition probability
        transition_score = transition_probability.loc[
            source_class,
            target_class
        ]

        # Combined score
        score = (
            suitability_score
            * transition_score
        )

        # Hanya pixel dari source class yang menjadi kandidat
        candidate_mask = source_mask

        candidate_scores = score[candidate_mask]

        candidate_info[
            (source_class, target_class)
        ] = {
            "quota": int(quota),
            "mask": candidate_mask,
            "scores": candidate_scores
        }

        print(
            f"  -> Target {target_class}: "
            f"quota = {quota:,}, "
            f"candidate pixels = {candidate_scores.size:,}"
        )

print("\n✓ Candidate preparation completed.")


# In[33]:


# ============================================================
# SPATIAL ALLOCATION - 2023 -> 2026
# ============================================================

for (source_class, target_class), info in candidate_info.items():

    quota = info["quota"]
    candidate_mask = info["mask"]
    candidate_scores = info["scores"]

    if quota == 0:
        continue

    print(
        f"Allocating Class {source_class} -> "
        f"Class {target_class}: {quota:,} pixels"
    )

    # Posisi pixel kandidat dalam raster
    candidate_positions = np.flatnonzero(
        candidate_mask.ravel()
    )

    # Urutkan berdasarkan score tertinggi
    # argpartition lebih hemat RAM daripada full sorting
    if quota < len(candidate_scores):

        selected_idx = np.argpartition(
            candidate_scores,
            -quota
        )[-quota:]

    else:

        selected_idx = np.arange(
            len(candidate_scores)
        )

    # Posisi raster yang dipilih
    selected_positions = candidate_positions[
        selected_idx
    ]

    # Ubah posisi 1D menjadi koordinat baris-kolom
    selected_rows, selected_cols = np.unravel_index(
        selected_positions,
        lc_2026.shape
    )

    # Pastikan pixel belum dialokasikan sebelumnya
    if np.any(
        allocated_mask[
            selected_rows,
            selected_cols
        ]
    ):
        raise ValueError(
            "Terdapat pixel yang sudah dialokasikan sebelumnya."
        )

    # Terapkan perubahan kelas
    lc_2026[
        selected_rows,
        selected_cols
    ] = target_class

    # Tandai sebagai sudah dialokasikan
    allocated_mask[
        selected_rows,
        selected_cols
    ] = True

    print(
        f"  ✓ Allocated: {len(selected_positions):,} pixels"
    )

print("\n=== SPATIAL ALLOCATION COMPLETED ===")
print(
    "Total allocated pixels:",
    allocated_mask.sum()
)


# In[34]:


# ============================================================
# VALIDATE PROJECTED LAND-COVER - 2026
# ============================================================

# Hitung jumlah pixel setiap kelas
class_counts_2026 = (
    pd.Series(lc_2026.ravel())
    .value_counts()
    .reindex(classes)
    .fillna(0)
    .astype(np.int64)
)

# Buat tabel perbandingan
validation_2026 = pd.DataFrame({
    "Classvalue": classes,
    "Classname": [
        class_table.loc[
            class_table["Classvalue"] == c,
            "Classname"
        ].iloc[0]
        for c in classes
    ],
    "Projected_2026_Markov": projected_counts_2026,
    "Spatial_Allocation_2026": class_counts_2026.values
})

validation_2026["Difference"] = (
    validation_2026["Spatial_Allocation_2026"]
    - validation_2026["Projected_2026_Markov"]
)

print("=== VALIDATION PROJECTED LAND-COVER 2026 ===")
display(validation_2026)

print("\nTotal pixels:", lc_2026.size)

print(
    "\nTotal difference:",
    validation_2026["Difference"].sum()
)

if np.all(validation_2026["Difference"] == 0):
    print("\n✓ 2026 SPATIAL ALLOCATION MATCHES MARKOV PROJECTION")
else:
    print("\n✗ THERE IS A DIFFERENCE — CHECK ALLOCATION")


# In[35]:


# ============================================================
# EXPORT PROJECTED LAND-COVER - 2026
# ============================================================

OUTPUT_2026 = OUTPUT_DIR / "Sarbagita_LandCover_2026.tif"

with rasterio.open(RASTER_2023) as src:

    profile = src.profile.copy()

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(
        OUTPUT_2026,
        "w",
        **profile
    ) as dst:

        dst.write(
            lc_2026.astype(np.uint8),
            1
        )

print("=== EXPORT 2026 ===")
print("Output:", OUTPUT_2026)
print("File exists:", OUTPUT_2026.exists())
print("File size (MB):", round(
    OUTPUT_2026.stat().st_size / (1024 * 1024),
    2
))


# In[36]:


# ============================================================
# VERIFY EXPORTED 2026 GEOTIFF
# ============================================================

with rasterio.open(OUTPUT_2026) as src:

    exported_2026 = src.read(1)

    print("=== EXPORTED 2026 GEOTIFF ===")
    print("File        :", OUTPUT_2026)
    print("Shape       :", exported_2026.shape)
    print("CRS         :", src.crs)
    print("Resolution  :", src.res)
    print("Dtype       :", src.dtypes[0])
    print("NoData      :", src.nodata)
    print("Bounds      :", src.bounds)

# Cek jumlah pixel setiap kelas
exported_counts_2026 = (
    pd.Series(exported_2026.ravel())
    .value_counts()
    .reindex(classes)
    .fillna(0)
    .astype(np.int64)
)

print("\n=== EXPORTED 2026 CLASS COUNTS ===")

display(
    exported_counts_2026.rename(
        "Pixel Count"
    ).to_frame()
)

print(
    "\nTotal pixels:",
    exported_2026.size
)

if np.array_equal(
    exported_counts_2026.values,
    projected_counts_2026
):
    print("\n✓ EXPORTED 2026 RASTER VERIFIED")
else:
    print("\n✗ EXPORTED RASTER DOES NOT MATCH PROJECTION")


# In[37]:


# ============================================================
# GEODESIC AREA CALCULATION FUNCTION
# ============================================================

geod = Geod(ellps="WGS84")


def calculate_class_area(raster_path, class_values):
    """
    Menghitung luas setiap kelas dalam hektare (ha)
    menggunakan geodesic area pada grid EPSG:4326.
    """

    with rasterio.open(raster_path) as src:

        raster = src.read(1)

        pixel_width = src.transform.a
        pixel_height = abs(src.transform.e)

        lon_left = src.transform.c
        lat_top = src.transform.f

        # Luas satu pixel untuk setiap baris
        row_pixel_area_m2 = np.zeros(
            src.height,
            dtype=np.float64
        )

        for row in range(src.height):

            lon_right = (
                lon_left + pixel_width
            )

            lat_bottom = (
                lat_top
                - (row + 1) * pixel_height
            )

            lat_row_top = (
                lat_top
                - row * pixel_height
            )

            polygon_lon = [
                lon_left,
                lon_right,
                lon_right,
                lon_left
            ]

            polygon_lat = [
                lat_row_top,
                lat_row_top,
                lat_bottom,
                lat_bottom
            ]

            area, _ = geod.polygon_area_perimeter(
                polygon_lon,
                polygon_lat
            )

            row_pixel_area_m2[row] = abs(area)

        results = []

        for class_value in class_values:

            mask = raster == class_value

            rows = np.where(mask)[0]

            area_m2 = np.sum(
                row_pixel_area_m2[rows]
            )

            area_ha = area_m2 / 10000

            results.append({
                "Classvalue": class_value,
                "Area_ha": area_ha
            })

        return pd.DataFrame(results)


# In[38]:


# ============================================================
# LAND-COVER AREA BY CLASS
# 2020, 2023, 2026
# ============================================================

area_2020 = calculate_class_area(
    OUTPUT_2020,
    classes
)

area_2023 = calculate_class_area(
    OUTPUT_2023,
    classes
)

area_2026 = calculate_class_area(
    OUTPUT_2026,
    classes
)

# Tambahkan nama kelas
for df in [area_2020, area_2023, area_2026]:

    df["Classname"] = [
        class_table.loc[
            class_table["Classvalue"] == c,
            "Classname"
        ].iloc[0]
        for c in df["Classvalue"]
    ]

# Gabungkan menjadi satu tabel
area_comparison = (
    area_2020[
        ["Classvalue", "Classname", "Area_ha"]
    ]
    .rename(columns={"Area_ha": "Area_2020_ha"})
    .merge(
        area_2023[
            ["Classvalue", "Area_ha"]
        ].rename(columns={"Area_ha": "Area_2023_ha"}),
        on="Classvalue"
    )
    .merge(
        area_2026[
            ["Classvalue", "Area_ha"]
        ].rename(columns={"Area_ha": "Area_2026_ha"}),
        on="Classvalue"
    )
)

# Perubahan luas 2023 -> 2026
area_comparison["Change_2023_2026_ha"] = (
    area_comparison["Area_2026_ha"]
    - area_comparison["Area_2023_ha"]
)

print("=== LAND-COVER AREA COMPARISON ===")

display(
    area_comparison.round(2)
)

print("\n=== TOTAL STUDY AREA ===")
print(
    f"2020 : {area_2020['Area_ha'].sum():,.2f} ha"
)
print(
    f"2023 : {area_2023['Area_ha'].sum():,.2f} ha"
)
print(
    f"2026 : {area_2026['Area_ha'].sum():,.2f} ha"
)


# In[39]:


# ============================================================
# LAND-COVER CHANGE ANALYSIS
# ============================================================

area_comparison["Change_2020_2023_ha"] = (
    area_comparison["Area_2023_ha"]
    - area_comparison["Area_2020_ha"]
)

area_comparison["Change_2020_2023_pct"] = (
    area_comparison["Change_2020_2023_ha"]
    / area_comparison["Area_2020_ha"]
    * 100
)

area_comparison["Change_2023_2026_pct"] = (
    area_comparison["Change_2023_2026_ha"]
    / area_comparison["Area_2023_ha"]
    * 100
)

# Susun kolom agar lebih mudah dibaca
area_change_table = area_comparison[
    [
        "Classvalue",
        "Classname",
        "Area_2020_ha",
        "Area_2023_ha",
        "Change_2020_2023_ha",
        "Change_2020_2023_pct",
        "Area_2026_ha",
        "Change_2023_2026_ha",
        "Change_2023_2026_pct"
    ]
].copy()

print("=== LAND-COVER CHANGE ANALYSIS ===")

display(
    area_change_table.round(2)
)


# In[40]:


# ============================================================
# ACCURACY ASSESSMENT SUMMARY
# ============================================================

accuracy_summary = accuracy_2020[
    ["Classvalue", "Classname",
     "Producer_Accuracy", "User_Accuracy"]
].copy()

accuracy_summary = accuracy_summary.rename(
    columns={
        "Producer_Accuracy": "Producer_Accuracy_2020",
        "User_Accuracy": "User_Accuracy_2020"
    }
)

accuracy_summary = accuracy_summary.merge(
    accuracy_2023[
        ["Classvalue",
         "Producer_Accuracy",
         "User_Accuracy"]
    ].rename(
        columns={
            "Producer_Accuracy": "Producer_Accuracy_2023",
            "User_Accuracy": "User_Accuracy_2023"
        }
    ),
    on="Classvalue"
)

accuracy_summary["Overall_Accuracy_2020"] = oa_2020
accuracy_summary["Cohen_Kappa_2020"] = kappa_2020

accuracy_summary["Overall_Accuracy_2023"] = oa_2023
accuracy_summary["Cohen_Kappa_2023"] = kappa_2023

# Ubah accuracy ke persen agar mudah dibaca
accuracy_display = accuracy_summary.copy()

accuracy_columns = [
    "Producer_Accuracy_2020",
    "User_Accuracy_2020",
    "Producer_Accuracy_2023",
    "User_Accuracy_2023",
    "Overall_Accuracy_2020",
    "Cohen_Kappa_2020",
    "Overall_Accuracy_2023",
    "Cohen_Kappa_2023"
]

accuracy_display[accuracy_columns] = (
    accuracy_display[accuracy_columns] * 100
)

print("=== ACCURACY ASSESSMENT SUMMARY ===")

display(
    accuracy_display.round(2)
)


# In[42]:


import sys
get_ipython().system('{sys.executable} -m pip install openpyxl')


# In[45]:


# ============================================================
# EXPORT ACCURACY AND LAND-COVER CHANGE TO EXCEL
# ============================================================

excel_output = OUTPUT_DIR / "Sarbagita_Accuracy_and_Change_Analysis.xlsx"

with pd.ExcelWriter(excel_output, engine="openpyxl") as writer:

    # Accuracy assessment
    accuracy_display.round(4).to_excel(
        writer,
        sheet_name="Accuracy_Assessment",
        index=False
    )

    # Land-cover area
    area_comparison.round(4).to_excel(
        writer,
        sheet_name="Area_Comparison",
        index=False
    )

    # Land-cover change
    area_change_table.round(4).to_excel(
        writer,
        sheet_name="Area_Change",
        index=False
    )

print("Excel berhasil dibuat:")
print(excel_output)

print("\nFile exists:", excel_output.exists())
print(
    "File size:",
    round(excel_output.stat().st_size / 1024, 2),
    "KB"
)


# In[46]:


# ============================================================
# EXPORT TRAINING SAMPLE SHAPEFILE
# ============================================================

training_output_dir = OUTPUT_DIR / "Training_Sample"
training_output_dir.mkdir(
    parents=True,
    exist_ok=True
)

training_output = (
    training_output_dir
    / "Sarbagita_Training_Sample.shp"
)

# Export training sample
training.to_file(
    training_output,
    driver="ESRI Shapefile"
)

print("Training sample berhasil diekspor:")
print(training_output)

# Cek file utama dan shapefile sidecar
required_files = [
    training_output,
    training_output.with_suffix(".shx"),
    training_output.with_suffix(".dbf"),
    training_output.with_suffix(".prj")
]

print("\n=== SHAPEFILE CHECK ===")

for file_path in required_files:
    print(
        file_path.name,
        "->",
        file_path.exists()
    )


# In[47]:


# ============================================================
# FINAL RESULTS SUMMARY
# ============================================================

final_summary = pd.DataFrame({
    "Classvalue": classes,
    "Classname": [
        class_table.loc[
            class_table["Classvalue"] == c,
            "Classname"
        ].iloc[0]
        for c in classes
    ],
    "Area_2020_ha": area_2020["Area_ha"].values,
    "Area_2023_ha": area_2023["Area_ha"].values,
    "Area_2026_ha": area_2026["Area_ha"].values
})

final_summary["Change_2020_2023_ha"] = (
    final_summary["Area_2023_ha"]
    - final_summary["Area_2020_ha"]
)

final_summary["Change_2023_2026_ha"] = (
    final_summary["Area_2026_ha"]
    - final_summary["Area_2023_ha"]
)

print("=== FINAL RESULTS SUMMARY ===")

display(
    final_summary.round(2)
)

print("\n=== MODEL ACCURACY ===")

print(
    f"2020 Overall Accuracy : {oa_2020:.4f}"
)
print(
    f"2020 Cohen's Kappa    : {kappa_2020:.4f}"
)
print(
    f"2023 Overall Accuracy : {oa_2023:.4f}"
)
print(
    f"2023 Cohen's Kappa    : {kappa_2023:.4f}"
)

print("\n=== TOTAL AREA ===")

print(
    f"2020 : {area_2020['Area_ha'].sum():,.2f} ha"
)

print(
    f"2023 : {area_2023['Area_ha'].sum():,.2f} ha"
)

print(
    f"2026 : {area_2026['Area_ha'].sum():,.2f} ha"
)


# In[48]:


# ============================================================
# CREATE README.md
# ============================================================

readme_path = OUTPUT_DIR / "README.md"

readme_text = f"""# Sarbagita Land-Cover Classification and 2026 Projection

## 1. Project Overview

This project was developed as part of the GIS Analyst case study for the Integrated Land Administration and Spatial Planning Project (ILAPPS).

The objective is to classify land cover in the Sarbagita Agglomeration area and generate a spatial land-cover projection for 2026.

The study area covers:

- Denpasar
- Badung
- Gianyar
- Tabanan
- Bali, Indonesia

The analysis uses RGB raster imagery for 2020 and 2023, training samples, Random Forest classification, transition probability analysis, Markov-based projection, and spatial allocation based on neighborhood suitability.

---

## 2. Land-Cover Classes

| Class Value | Land-Cover Class |
|-------------:|------------------|
| 1 | Pemukiman atau Lahan Terbangun |
| 21 | Lahan Kosong |
| 35 | Tubuh Air |
| 40 | Pertanian |
| 56 | Vegetasi |

---

## 3. Input Data

### Raster Data

- `Sarbagita_2020.tif`
- `Sarbagita_2023.tif`

Raster characteristics:

- CRS: EPSG:4326
- Spatial resolution: approximately 8.98e-05 degrees
- 3 bands
- RGB data
- Data type: uint8

### Training Data

Training samples were provided as a polygon shapefile and contain five land-cover classes.

---

## 4. Methodology

### 4.1 Training Sample Extraction

Training polygons were overlaid with the raster imagery to extract RGB pixel values.

The predictors used in the classification were:

- Red
- Green
- Blue

The supplied raster contains RGB bands only. Therefore, multispectral indices such as NDVI or NDWI were not calculated.

### 4.2 Random Forest Classification

Random Forest was used independently for the 2020 and 2023 classifications.

Model configuration:

- Number of trees: 200
- Random state: 42
- Class weighting: balanced
- Validation split: 20%
- Stratified sampling

### 4.3 Accuracy Assessment

Model performance was evaluated using:

- Overall Accuracy
- Cohen's Kappa
- Producer's Accuracy
- User's Accuracy

### 4.4 Land-Cover Transition Analysis

The classified 2020 and 2023 maps were compared pixel-by-pixel to generate a land-cover transition matrix.

Transition probabilities were calculated by dividing each transition count by the total number of pixels in the corresponding 2020 source class.

### 4.5 2026 Projection

A Markov transition approach was used to estimate the 2026 class totals based on the 2020–2023 transition probabilities.

The projected class totals were then spatially allocated using neighborhood-based suitability derived from the 2023 classification.

The allocation procedure:

1. Identify classes with projected gains and losses.
2. Calculate transition allocation quotas.
3. Generate neighborhood suitability surfaces using a 5 × 5 moving window.
4. Rank candidate pixels according to suitability.
5. Allocate the required number of pixels for each source-target transition.
6. Validate that source losses and target gains exactly match the projected totals.

---

## 5. Accuracy Results

| Metric | 2020 | 2023 |
|--------|-----:|-----:|
| Overall Accuracy | {oa_2020 * 100:.2f}% | {oa_2023 * 100:.2f}% |
| Cohen's Kappa | {kappa_2020:.4f} | {kappa_2023:.4f} |

### Class-Level Accuracy

| Class | Producer's Accuracy 2020 | User's Accuracy 2020 | Producer's Accuracy 2023 | User's Accuracy 2023 |
|-------|-------------------------:|----------------------:|-------------------------:|----------------------:|
"""

for _, row in accuracy_summary.iterrows():

    readme_text += (
        f"| {row['Classname']} | "
        f"{row['Producer_Accuracy_2020'] * 100:.2f}% | "
        f"{row['User_Accuracy_2020'] * 100:.2f}% | "
        f"{row['Producer_Accuracy_2023'] * 100:.2f}% | "
        f"{row['User_Accuracy_2023'] * 100:.2f}% |\\n"
    )

readme_text += f"""
---

## 6. Land-Cover Area Results

Area was calculated using geodesic pixel area on the original EPSG:4326 grid.

| Class | 2020 (ha) | 2023 (ha) | 2026 (ha) |
|-------|----------:|----------:|----------:|
"""

for _, row in final_summary.iterrows():

    readme_text += (
        f"| {row['Classname']} | "
        f"{row['Area_2020_ha']:,.2f} | "
        f"{row['Area_2023_ha']:,.2f} | "
        f"{row['Area_2026_ha']:,.2f} |\\n"
    )

readme_text += f"""
**Total study area:**

- 2020: {area_2020['Area_ha'].sum():,.2f} ha
- 2023: {area_2023['Area_ha'].sum():,.2f} ha
- 2026: {area_2026['Area_ha'].sum():,.2f} ha

---

## 7. Land-Cover Change

| Class | Change 2020–2023 (ha) | Change 2023–2026 (ha) |
|-------|----------------------:|----------------------:|
"""

for _, row in final_summary.iterrows():

    readme_text += (
        f"| {row['Classname']} | "
        f"{row['Change_2020_2023_ha']:,.2f} | "
        f"{row['Change_2023_2026_ha']:,.2f} |\\n"
    )

readme_text += """
---

## 8. Outputs

The following outputs are generated by the analysis:

- `Sarbagita_LandCover_2020.tif`
- `Sarbagita_LandCover_2023.tif`
- `Sarbagita_LandCover_2026.tif`
- `Sarbagita_Accuracy_and_Change_Analysis.xlsx`
- `Training_Sample/Sarbagita_Training_Sample.shp`
- `Training_Sample/Sarbagita_Training_Sample.shx`
- `Training_Sample/Sarbagita_Training_Sample.dbf`
- `Training_Sample/Sarbagita_Training_Sample.prj`

---

## 9. Limitations

1. The supplied imagery contains RGB bands only. Therefore, the analysis does not use full Sentinel-2 multispectral bands or spectral indices such as NDVI and NDWI.
2. The classification results depend on the quality and representativeness of the supplied training samples.
3. The lower class-level accuracy for some classes, particularly Lahan Kosong in 2023, indicates greater classification uncertainty for those classes.
4. The 2026 projection is a model-based spatial projection and should not be interpreted as a deterministic prediction of actual future land cover.
5. Transition probabilities are derived from the 2020–2023 classification results, so classification uncertainty can propagate into the projection.
6. Spatial allocation uses neighborhood suitability derived from the 2023 classification and does not incorporate external socioeconomic, planning, road-network, or policy datasets.

---

## 10. Reproducibility

The analysis was developed in Python using:

- Python
- NumPy
- Pandas
- GeoPandas
- Rasterio
- Scikit-learn
- PyProj
- OpenPyXL

Random Forest uses a fixed random state (`42`) to improve reproducibility.

All raster outputs preserve the original study-area extent and spatial reference.

---

## 11. Recommended Execution Order

1. Load input raster and training data.
2. Extract training pixels.
3. Train and validate the 2020 Random Forest model.
4. Generate the 2020 land-cover classification.
5. Train and validate the 2023 Random Forest model.
6. Generate the 2023 land-cover classification.
7. Calculate the 2020–2023 transition matrix and probabilities.
8. Generate the 2026 Markov projection.
9. Calculate neighborhood suitability.
10. Perform spatial allocation.
11. Validate projected class totals.
12. Export the 2026 raster.
13. Calculate land-cover areas and changes.
14. Export accuracy and change analysis to Excel.
15. Export the training sample shapefile.

"""

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_text)

print("README berhasil dibuat:")
print(readme_path)

print("\nFile exists:", readme_path.exists())
print(
    "File size:",
    round(readme_path.stat().st_size / 1024, 2),
    "KB"
)


# In[49]:


# ============================================================
# FINAL OUTPUT INVENTORY
# ============================================================

print("=== FINAL OUTPUT INVENTORY ===\n")

for path in sorted(OUTPUT_DIR.rglob("*")):

    if path.is_file():

        size_kb = path.stat().st_size / 1024

        print(
            f"[OK] {path.relative_to(OUTPUT_DIR)} "
            f"({size_kb:.2f} KB)"
        )

print("\n=== REQUIRED OUTPUT CHECK ===")

required_outputs = [
    OUTPUT_DIR / "Sarbagita_LandCover_2020.tif",
    OUTPUT_DIR / "Sarbagita_LandCover_2023.tif",
    OUTPUT_DIR / "Sarbagita_LandCover_2026.tif",
    OUTPUT_DIR / "Sarbagita_Accuracy_and_Change_Analysis.xlsx",
    OUTPUT_DIR / "README.md",
    OUTPUT_DIR / "Training_Sample" / "Sarbagita_Training_Sample.shp",
    OUTPUT_DIR / "Training_Sample" / "Sarbagita_Training_Sample.shx",
    OUTPUT_DIR / "Training_Sample" / "Sarbagita_Training_Sample.dbf",
    OUTPUT_DIR / "Training_Sample" / "Sarbagita_Training_Sample.prj"
]

all_ok = True

for path in required_outputs:

    exists = path.exists()

    print(
        f"{'✓' if exists else '✗'} "
        f"{path.relative_to(OUTPUT_DIR)}"
    )

    if not exists:
        all_ok = False

print("\n=== FINAL STATUS ===")

if all_ok:
    print("✓ ALL REQUIRED OUTPUTS ARE COMPLETE")
else:
    print("✗ SOME REQUIRED OUTPUTS ARE MISSING")


# In[50]:


# ============================================================
# EXPORT FINAL NOTEBOOK TO PYTHON SCRIPT
# ============================================================

from pathlib import Path
import nbformat
from nbconvert import PythonExporter

# Final notebook
notebook_path = Path(
    "/mnt/d/ILAPS/ILAPPS_GIS_Analyst_Final.ipynb"
)

# Final Python script
script_path = Path(
    "/mnt/d/ILAPS/outputs/Sarbagita_LandCover_Projection.py"
)

# Check notebook
if not notebook_path.exists():
    raise FileNotFoundError(
        f"Notebook tidak ditemukan: {notebook_path}"
    )

# Read notebook
with open(
    notebook_path,
    "r",
    encoding="utf-8"
) as f:

    notebook = nbformat.read(
        f,
        as_version=4
    )

# Convert notebook to Python
exporter = PythonExporter()

script_content, _ = (
    exporter.from_notebook_node(notebook)
)

# Save Python script
with open(
    script_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(script_content)

print("✓ Python script berhasil dibuat:")
print(script_path)

print("\nFile exists:", script_path.exists())

print(
    "File size:",
    round(
        script_path.stat().st_size / 1024,
        2
    ),
    "KB"
)


# In[ ]:




