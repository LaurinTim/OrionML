import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter
import pprint
pp = pprint.PrettyPrinter(indent=4)

import joblib
from skimage.io import imread
from skimage.transform import resize, rescale
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import skimage
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

data_path = r'C:\Users\main\Proton Drive\laurin.koller\My files\ML\repos\OrionML\Examples\example data\animal_face'

import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir("C:\\Users\\main\\Proton Drive\\laurin.koller\\My files\\ML\\repos\\OrionML")

import OrionML as orn

# %%

def resize_all(src, pklname, include, width=150, height=None):
    """
    load images from path, resize them and write them as arrays to a dictionary, 
    together with labels and metadata. The dictionary is written to a pickle file 
    named '{pklname}_{width}x{height}px.pkl'.
     
    Parameter
    ---------
    src: str
        path to data
    pklname: str
        path to output file
    width: int
        target width of the image in pixels
    include: set[str]
        set containing str
    """
    
    height = height if height is not None else width
     
    data = dict()
    data['description'] = 'resized ({0}x{1})animal images in rgb'.format(int(width), int(height))
    data['label'] = []
    data['filename'] = []
    data['data'] = []   
     
    pklname = f"{pklname}_{width}x{height}px.pkl"
 
    # read all images in PATH, resize and write to DESTINATION_PATH
    for subdir in os.listdir(src):
        if subdir in include:
            print(subdir)
            current_path = os.path.join(src, subdir)
 
            for file in os.listdir(current_path):
                if file[-3:] in {'jpg', 'png'}:
                    im = imread(os.path.join(current_path, file))
                    im = resize(im, (width, height)) #[:,:,::-1]
                    data['label'].append(subdir[:-4])
                    data['filename'].append(file)
                    data['data'].append(im)
 
        joblib.dump(data, pklname)
        
base_name = r"C:\Users\main\Desktop\ML\AnimalFace\pkl_files\animal_faces"
width = 80
 
include = {'ChickenHead', 'BearHead', 'ElephantHead', 'EagleHead', 
           'DeerHead', 'MonkeyHead', 'PandaHead', 'CatHead', 'CowHead', 
           'DuckHead', 'LionHead', 'HumanHead'}
 
#resize_all(src=data_path + "\\Image", pklname=base_name, width=width, include=include)

# %%

data = joblib.load(f'{base_name}_{width}x{width}px.pkl')
 
print('number of samples: ', len(data['data']))
print('keys: ', list(data.keys()))
print('description: ', data['description'])
print('image shape: ', data['data'][0].shape)
print('labels:', np.unique(data['label']))
 
#Counter(data['label'])

# use np.unique to get all unique values in the list of labels
labels = np.unique(data['label'])

# %%
 
# set up the matplotlib figure and axes, based on the number of labels
fig, axes = plt.subplots(1, len(labels))
fig.set_size_inches(15,4)
fig.tight_layout()
 
# make a plot for every label (equipment) type. The index method returns the 
# index of the first item corresponding to its search string, label in this case
for ax, label in zip(axes, labels):
    idx = data['label'].index(label)
     
    ax.imshow(data['data'][idx])
    ax.axis('off')
    ax.set_title(label)
plt.show()

# %%

for i in range(len(data['data'])):
    if data['data'][i].shape != (80,80,3):
        print(i, data['data'][i].shape)
        plt.imshow(data['data'][i-1])

# %%
    
X = np.array(data['data'])
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True,
    random_state=42
)

def plot_bar(y, loc='left', relative=True):
    width = 0.35
    if loc == 'left':
        n = -0.5
    elif loc == 'right':
        n = 0.5
     
    # calculate counts per type and sort, to ensure their order
    unique, counts = np.unique(y, return_counts=True)
    sorted_index = np.argsort(unique)
    unique = unique[sorted_index]
     
    if relative:
        # plot as a percentage
        counts = 100*counts[sorted_index]/len(y)
        ylabel_text = '% count'
    else:
        # plot counts
        counts = counts[sorted_index]
        ylabel_text = 'count'
         
    xtemp = np.arange(len(unique))
     
    plt.bar(xtemp + n*width, counts, align='center', alpha=.7, width=width)
    plt.xticks(xtemp, unique, rotation=45)
    plt.xlabel('equipment type')
    plt.ylabel(ylabel_text)
 
plt.suptitle('relative amount of photos per type')
plot_bar(y_train, loc='left')
plot_bar(y_test, loc='right')
plt.legend([
    'train ({0} photos)'.format(len(y_train)), 
    'test ({0} photos)'.format(len(y_test))
]);
plt.show()

# %%

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])

class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
        
        return np.array([local_hog(img) for img in X])
        
        #try: # parallel
        #    return np.array([local_hog(img) for img in X])
        #except:
        #    return np.array([local_hog(img) for img in X])

# %%

# create an instance of each transformer
grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(10, 10), 
    cells_per_block=(3,3), 
    orientations=10, 
    block_norm='L2-Hys'
)
scalify = StandardScaler()

# call fit_transform on each transform converting X_train step by step
X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, loss='hinge', alpha=0.0001, n_jobs=-1, learning_rate="optimal", random_state=13)
sgd_clf.fit(X_train_prepared, y_train)

X_test_gray = grayify.transform(X_test)
X_test_hog = hogify.transform(X_test_gray)
X_test_prepared = scalify.transform(X_test_hog)

y_pred = sgd_clf.predict(X_test_prepared)

print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))

# %%

def arr_one(pos):
    arr = np.zeros(len(labels))
    arr[pos] = 1
    return arr

one_hot_map = {labels[i]: arr_one(i) for i in range(len(labels))}

y_traino = np.array([one_hot_map.get(val) for val in y_train])
y_testo = np.array([one_hot_map.get(val) for val in y_test])

# %%

gd = orn.method.GDClassifier(alpha=1e-2, num_iters=1000, verbose=True)
gd.fit(X_train_prepared, y_traino)

# %%

y_predo = gd.predict(X_test_prepared)

print('Percentage correct: ', 100*np.sum([(val==bal).all() for val,bal in zip(y_predo,y_testo)])/len(y_testo))

# %%

cmx = confusion_matrix(y_test, y_pred)
 
def plot_confusion_matrix(cmx, vmax1=None, vmax2=None, vmax3=None):
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    cmx_norm = np.around(cmx_norm, 1)
    cmx_zero_diag = cmx_norm.copy()
 
    np.fill_diagonal(cmx_zero_diag, 0)
 
    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(12, 6)
    [a.set_xticks(range(len(labels)), labels=labels, rotation=45, size=12, ha='right', rotation_mode="anchor") for a in ax]
    [a.set_yticks(range(len(labels)), labels=labels, size=12) for a in ax]
    [a.set_xlabel(xlabel="Predicted Label", size=12) for a in ax]
    [a.set_ylabel(ylabel="True Label", size=12) for a in ax]
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cmx_norm[i, j]>=np.max(cmx_norm)*0.6:
                ax[0].text(j, i, cmx_norm[i, j], ha="center", va="center", c="black", size=6)
            else:
                ax[0].text(j, i, cmx_norm[i, j], ha="center", va="center", c="white", size=6)
                
            if cmx_zero_diag[i, j]>=np.max(cmx_zero_diag)*0.5:
                ax[1].text(j, i, cmx_zero_diag[i, j], ha="center", va="center", c="black", size=6)
            else:
                ax[1].text(j, i, cmx_zero_diag[i, j], ha="center", va="center", c="white", size=6)
         
    im1 = ax[0].imshow(cmx_norm, vmax=vmax2)
    ax[0].set_title('%')
    im2 = ax[1].imshow(cmx_zero_diag, vmax=vmax3)
    ax[1].set_title('% and 0 diagonal')
 
    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2 = [divider.append_axes("right", size="5%", pad=0.1) 
                        for divider in dividers]
 
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.tight_layout()
     
plot_confusion_matrix(cmx)
plt.show()

# %%

cmx = confusion_matrix(y_test, y_pred)
 
def plot_confusion_matrix(cmx, vmax1=None, vmax2=None, vmax3=None):
    cmx_norm = 100*cmx / cmx.sum(axis=1, keepdims=True)
    cmx_zero_diag = cmx_norm.copy()
 
    np.fill_diagonal(cmx_zero_diag, 0)
 
    fig, ax = plt.subplots(ncols=3)
    fig.set_size_inches(12, 3)
    [a.set_xticks(range(len(cmx)+1)) for a in ax]
    [a.set_yticks(range(len(cmx)+1)) for a in ax]
         
    im1 = ax[0].imshow(cmx, vmax=vmax1)
    ax[0].set_title('as is')
    im2 = ax[1].imshow(cmx_norm, vmax=vmax2)
    ax[1].set_title('%')
    im3 = ax[2].imshow(cmx_zero_diag, vmax=vmax3)
    ax[2].set_title('% and 0 diagonal')
 
    dividers = [make_axes_locatable(a) for a in ax]
    cax1, cax2, cax3 = [divider.append_axes("right", size="5%", pad=0.1) 
                        for divider in dividers]
 
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im3, cax=cax3)
    fig.tight_layout()
     
plot_confusion_matrix(cmx)
plt.show()
































