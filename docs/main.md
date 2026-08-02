# Main API

::: model.main
    handler: python
    options:
        docstring_style: google
        show_root_heading: true
        show_source: true
        merge_init_into_class: true
        members:
            - predict
            - read_predictions
            - load_data_from_upload
            - predict_with_model