Nifti Image Translation API

This project is a Flask-based backend API for an image translation application.


* What It Does

- Accepts NIfTI medical images (.nii, .nii.gz) via HTTP.
- Translates them into a target modality (e.g. from CT to PET) using a pre-trained generative model.
- Returns the generated image as a NIfTI file.

* Progress Feedback
  - By modifying the inference code to print remaining iterations to stdout, the UI can poll and display progress dynamically.


* To customize or expand the models available to the UI:

- Add more checkpoints to the designated directory `Checkpoints`.
- Update the models.json file as such
```json
[
  {
    "id": 1,
    "title": "CT-to-PET (CL_ff)",
    "description": "Converts CT scans to synthetic PET images. Trained with curriculum learning with a forgetting factor",
    "inputModality": "CT",
    "outputModality": "PET",
    "region": "Total Body",
    "modelPath": "CL_ff_0.8_v2",
    "networkName": "BEST_final_400"
  }
]
```

  - Id should be unique, set it to a value not existing in the other models
  - Title and description is displayed in the UI
  - Input and output modality is set to indicate what modlities the translation model is for
  - region is to specify the intended translated region, or what the model has been trained on.
  - modelPath is the directory to the checkpoint inside the `Chekcpoints` folder
  - networkName is the name of the network

These changes allow you to dynamically add or remove models exposed through the API.
