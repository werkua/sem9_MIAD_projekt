import pandas as pd
#import h5py
import numpy as np
import matplotlib.pyplot as plt
files = ["Data/atm_muon.h5",
         "Data/atm_neutrino_classA.h5",
         "Data/atm_neutrino_classB.h5",
         "Data/atm_neutrino_classC.h5",
         "Data/atm_neutrino_classD.h5",
         "Data/atm_neutrino_classE.h5",
         "Data/atm_neutrino_classF.h5",
         "Data/atm_neutrino_classG.h5",
         "Data/atm_neutrino_classH.h5"
        ]

# files for tracks: classB classF
dfB = pd.read_hdf(files[2])
# dfF = pd.read_hdf(files[6])

# for showers: classA classC classE classG
dfA = pd.read_hdf(files[1])
# dfC = pd.read_hdf(files[3])
# dfE = pd.read_hdf(files[5])
# dfG = pd.read_hdf(files[7])

# # tracks: 1
class1 = [1]*len(dfB)
dfB["label"] = class1
# # showers: 0
class0 = [0]*len(dfB)
dfA["label"] = class0

dfAB = pd.concat([dfA, dfB])
XY = dfAB.sample(frac=1).reset_index(drop=True)


dirR = np.sqrt((XY["dir_x"])**2 + (XY["dir_y"])**2 + (XY["dir_z"])**2)
# XY["dir_r"] = dirR
# plot dirR 3D plot
plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(XY["dir_x"], XY["dir_y"], dirR, c=dirR, cmap='viridis', s=1)
ax.set_xlabel('dir_x')
ax.set_ylabel('dir_y')
ax.set_zlabel('dirR')
plt.title('3D Scatter Plot of Direction Vectors Colored by Magnitude')
plt.savefig("plots/3d_dir_vectors.pdf")
plt.show()
plt.close()

# X = XY.drop(columns=["dir_x", "dir_y", "dir_z",'label'])
# print(X)
# print("\n")
# Y = XY[['label']]
# print(Y)


#------------------------------------------------------
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_repor

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# y_pred = rf_classifier.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)

# print(f"Accuracy: {accuracy:.2f}")
# print("\nClassification Report:\n", classification_rep)