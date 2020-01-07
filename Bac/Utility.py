import keras


# Save the model
def serialize_model(model, path="latest.h5"):
    model.save(path)


# Recreate the exact same model purely from the file
def deserialize_model(path="latest.h5"):
    return keras.models.load_model(path)