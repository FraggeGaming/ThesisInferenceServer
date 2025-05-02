Nifti Image Translation API
This project is a Flask-based backend API for an image translation application.

What It Does
Accepts NIfTI medical images (.nii, .nii.gz) via HTTP.

Translates them into a target modality (e.g. from CT to PET) using a pre-trained generative model.

Returns the generated image as a NIfTI file.

Progress Feedback
By modifying the inference code to print remaining iterations to stdout, the UI can poll and display progress dynamically.

Modifications
To customize or expand the models available to the UI:
Add more checkpoints to the designated directory.
Change the checkpoint path in the configuration or inference code.
Import alternate inference subprocesses to support different models or modalities.
These changes allow you to dynamically add or remove models exposed through the API.
