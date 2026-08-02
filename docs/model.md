# Model Training

Module d'entraînement du modèle de prédiction.

::: model.model
    handler: python
    options:
        docstring_style: google
        show_root_heading: true
        show_source: true
        merge_init_into_class: true
        members:
            - load_and_prepare_data
            - create_model
            - create_param_grid
            - train_model
            - save_model
            - main