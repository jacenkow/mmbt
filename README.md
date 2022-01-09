# Indication as Prior Knowledge for Multimodal Disease Classification in Chest Radiographs with Transformers

Code for the paper:

> Jacenków, G., O’Neil, A.Q. and Tsaftaris, S.A., 2022, March. Indication as
> Prior Knowledge for Multimodal Disease Classification in Chest Radiographs
> with Transformers. In 2022 IEEE 19th International Symposium on Biomedical
> Imaging (ISBI). IEEE.

### Requirements

The project was developed in Python 3.7.9, PyTorch 1.8.2 with CUDA 11.1
acceleration. We use the [MMF framework](https://mmf.sh/), which provides a
boilerplate code for data loaders and common visual-linguistic models. Please,
follow the MMF website on how to install the framework.

We use the [MIMIC-CXR dataset](https://physionet.org/content/mimic-cxr/2.0.0/)
for our experiments. Due to the licensing agreement, we cannot share the images
and radiology reports. Please, refer to the PhysioNet website for details on how
to acquire the dataset. We recommend downloading the MIMIC-CXR-JPG dataset to
avoid additional pre-processing of DICOM images.

### Setup

We expect the images in JPG format to be placed in `dataset/mimic_cxr/subjects`
folder. Please, follow the folder to see an example. All textual information,
i.e., corresponding radiology reports, ground truth labels, subject and study
IDs, are stored as `jsonl` files in the `dataset/mimic_cxr/annotations` folder.
We expect three files to run the experiments, i.e., `test.jsonl`,
`training.jsonl`, and `val.jsonl`. Please, see the example.

### Tools

We add several tools to ease the preprocessing and training processes, i.e.,

```
* image_preprocessing.py - resize all images to 224 x 224 size.
* mimic_cxr_preprocess.py - extract indication/history fields from the full radiology reports.
* precalculate_embeddings.py - precalculate BioWordVec embeddings to avoid bottlenecking during training.
```

### First run

Please, follow the commands to run the experiments (training and evaluation).

```
pip install -r requirements.txt  # run once, after installing the MMF framework.
pip install -e .  # run once, install the project as a Python package.
./run.sh  # run training and evaluation protocol.
```

### Evaluation

The evaluation protocol will store predictions as `json` files. Please, use
the Jupyter notebooks to evaluate the predictions with the aforementioned files.
