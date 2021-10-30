from pipeline import Pipeline


def pipeline_model(model_class):
    def model_init(*args, **kwargs):

        if not Pipeline._current_pipeline_defining:
            return model_class(*args, **kwargs)
        else:
            created_model = model_class(*args, **kwargs)

            model_functions = [
                model_attr
                for model_attr in dir(created_model)
                if callable(getattr(created_model, model_attr))
                and model_attr[:2] != "__"
            ]
            print(model_functions)
            return created_model

    model_class.__pipeline_model__ = True
    return model_init
