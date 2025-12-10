import pandas as pd
#import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D  # needed to enable 3D projection
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
xs = ["pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z", "time", "tot"]
# for i in files:
#     df = pd.read_hdf(i)
#     print(f"File: {i} has {len(df)} entries.")
#     print(df.head())
# df1 = pd.read_hdf("Data/atm_muon.h5")
# # print(df1.head())


def narysuj_histogram(dfs, column_i, nobins=50):
    plt.figure()
    for i in range(len(dfs)):
        df = dfs[i]
        x = df[xs[column_i]]
        plt.hist(x, bins=nobins, alpha=0.5, label=f'File {i}')
    plt.legend()
    plt.yscale('log')
    plt.savefig("plots/histogram_" + x.name + ".pdf")

    
# narysuj_histogram(df1['pos_x'])
# narysuj_histogram(df1['pos_y'])
# narysuj_histogram(df1['pos_z'])
# narysuj_histogram(df1['dir_x'])
# narysuj_histogram(df1['dir_y'])
# narysuj_histogram(df1['dir_z'])
# narysuj_histogram(df1['time'])
# narysuj_histogram(df1['tot'])
dfs = []
for i in files:
    df = pd.read_hdf(i)
    dfs.append(df)
for i in range(len(df.columns)):
        narysuj_histogram(dfs, i, 50)
    

#-----------------------------------------------------
# def plot_3d_positions(h5_file, n_sample=None, color='C0', s=1, alpha=0.6, out_dir="plots"):
#     os.makedirs(out_dir, exist_ok=True)
#     df = pd.read_hdf(h5_file)                      # load HDF5 into a pandas DataFrame
#     # Select columns and convert to numpy
#     coords = df[['pos_x', 'pos_y', 'pos_z']]
#     if n_sample is not None and len(coords) > n_sample:
#         coords = coords.sample(n=n_sample, random_state=0)
#     x = coords['pos_x'].values
#     y = coords['pos_y'].values
#     z = coords['pos_z'].values

#     fig = plt.figure(figsize=(8,6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x, y, z, c=color, s=s, alpha=alpha)
#     ax.set_xlabel('pos_x')
#     ax.set_ylabel('pos_y')
#     ax.set_zlabel('pos_z')
#     basename = os.path.splitext(os.path.basename(h5_file))[0]
#     out_path = os.path.join(out_dir, f"3d_{basename}.pdf")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close(fig)
#     print(f"Saved 3D scatter to {out_path}")

# # Example usage: create one plot per file, sampling up to 100k points
# for i, fpath in enumerate(files):
#     plot_3d_positions(fpath, n_sample=100000, color=f"C{i%10}", s=1, alpha=0.6)
    
def plot_3d_positions_with_vectors(
    h5_file,
    n_sample=None,         # how many points to scatter (None = all)
    n_arrows=500,          # how many arrows to draw (sampled)
    arrow_scale=0.1,       # base arrow length scaling factor
    normalize_arrows=False,# if True, arrows have unit length then scaled by arrow_scale
    color='C0',
    s=1,
    alpha=0.6,
    out_dir="plots"
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_hdf(h5_file)

    # Basic checks
    for col in ['pos_x','pos_y','pos_z','dir_x','dir_y','dir_z']:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {h5_file}")

    coords = df[['pos_x', 'pos_y', 'pos_z']]

    if n_sample is not None and len(coords) > n_sample:
        coords_scatter = coords.sample(n=n_sample, random_state=0)
    else:
        coords_scatter = coords

    x = coords_scatter['pos_x'].values
    y = coords_scatter['pos_y'].values
    z = coords_scatter['pos_z'].values

    # Prepare arrows (sample separately so arrows are sparser)
    if n_arrows is None or n_arrows <= 0:
        arrow_sample = pd.DataFrame(columns=['pos_x','pos_y','pos_z','dir_x','dir_y','dir_z'])
    else:
        if len(df) > n_arrows:
            arrow_sample = df.sample(n=n_arrows, random_state=1)
        else:
            arrow_sample = df

    xa = arrow_sample['pos_x'].values
    ya = arrow_sample['pos_y'].values
    za = arrow_sample['pos_z'].values
    ua = arrow_sample['dir_x'].values
    va = arrow_sample['dir_y'].values
    wa = arrow_sample['dir_z'].values

    # Optionally normalize direction vectors to unit length before scaling
    if normalize_arrows:
        norms = np.linalg.norm(np.vstack([ua,va,wa]).T, axis=1)
        # avoid division by zero
        norms[norms == 0] = 1.0
        ua = ua / norms
        va = va / norms
        wa = wa / norms

    # Scale arrows so they are visible; length param multiplies vector lengths
    # You can tune arrow_scale to taste
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=color, s=s, alpha=alpha)
    if len(xa) > 0:
        ax.quiver(xa, ya, za, ua, va, wa, length=arrow_scale, normalize=False, linewidth=0.5, color='k', alpha=0.8)

    ax.set_xlabel('pos_x')
    ax.set_ylabel('pos_y')
    ax.set_zlabel('pos_z')
    basename = os.path.splitext(os.path.basename(h5_file))[0]
    out_path = os.path.join(out_dir, f"3d_vectors_{basename}.pdf")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved 3D scatter+vectors to {out_path}")
    
plot_3d_positions_with_vectors("Data/atm_muon.h5", n_sample=10000, n_arrows=100, arrow_scale=5)