import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file
import os
import cv2

path = "train\\"

def show(ds):
    fpath = get_testdata_file('CT_small.dcm')
    ds = dcmread(fpath)

    # Normal mode:
    print()
    print(f"SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
    print()

    pat_name = ds.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print(f"Patient's Name...: {display_name}")
    print(f"Patient ID.......: {ds.PatientID}")
    print(f"Modality.........: {ds.Modality}")
    print(f"Study Date.......: {ds.StudyDate}")
    print(f"Image size.......: {ds.Rows} x {ds.Columns}")

    # use .get() if not sure the item exists, and want a default value if missing
    print(f"Slice location...: {ds.get('SliceLocation', '(missing)')}")

    # plot the image using matplotlib
    plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
    plt.show()

#debug: for debug
#reshape: for reshape, ie (100,100)
def readall(**kwargs):
    files = os.listdir(path)
    if 'debug' in kwargs:
        print(files)
    for file in files:
        yield read(file,**kwargs)

def tf_read_all(shape = (224,224)):
    files = os.listdir(path)
    for file in files:
        yield read(file,reshape=shape)

def tf_data(SHAPE = (224,224)):
    from tensorflow.data.Dataset import from_generator
    from tensorflow import int32 
    
    return from_generator(tf_read_all,args=[SHAPE],output_types= int32,output_shapes = SHAPE+(3))

#debug: for debug
#reshape: for reshape, ie (100,100)
def read(file_path, **kwargs):
    ds = dcmread(path + file_path)
    if 'debug' in kwargs:
        print(ds.pixel_array.shape)
    if 'reshape' in kwargs:
        return resize(ds.pixel_array,kwargs['shape'])
    return ds.pixel_array

def resize(img, new_shape):
    return cv2.resize(img, dsize=new_shape, interpolation=cv2.INTER_CUBIC)


if __name__ == "__main__":
    fpath = get_testdata_file('CT_small.dcm')
    print(fpath)
    fpath = "train/ISIC_0015719.dcm"
    ds = dcmread(fpath)
    show(ds)