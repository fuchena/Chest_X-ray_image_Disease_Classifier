from keras_preprocessing.image import ImageDataGenerator


class Generator:

    def __init__(self, model, img_weight, img_height,batch_size):
        self.model = model
        self.img_weight = img_weight
        self.img_height = img_height
        self.batch_size = batch_size

    def generator_path(self, train_data_path, validation_data_path, nb_train_samples, epochs,nb_validation_samples):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            train_data_path,
            target_size=(self.img_height, self.img_weight),
            batch_size=self.batch_size,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            validation_data_path,
            target_size=(self.img_height, self.img_weight),
            batch_size=self.batch_size,
            class_mode='categorical')

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples)
        return self.model


