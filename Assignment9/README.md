- **Assignment9.ipynb**: This Jupyter Notebook file contains the full implementation of the project.
- **model.py**: This Python file contains the implementation of the neural network.

## Argumentation

The data augmentation techniques used in this project are as follows:

- **HorizontalFlip (p=0.5)**
- **ShiftScaleRotate**: This augmentation randomly shifts, scales, and rotates the input images. The parameters used are:
  - shift_limit=0.1
  - scale_limit=0.1
  - rotate_limit=10
- **CoarseDropout**: This augmentation technique applies dropout by removing rectangular chunks from the input images. The parameters used are:
  - max_holes=1
  - max_height=16
  - max_width=16
  - min_holes=1
  - min_height=16
  - min_width=16
  - fill_value=mean

## Model Details

- **Total Parameters**: 227,424
- **RF (Receptive Field)**: 43
