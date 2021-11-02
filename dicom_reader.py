import matplotlib.pyplot as plt
from pydicom import dcmread
from pydicom.data import get_testdata_file

fpath = get_testdata_file('CT_small.dcm')
ds = dcmread(fpath)

def show(ds):
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
        
if __name__ == "__main__":
    fpath = get_testdata_file('CT_small.dcm')
    fpath = "train\ISIC_0015719.dcm"
    ds = dcmread(fpath)
    show(ds)