from classifier import BinaryClassifier, ImageIngestion, MakePredictions

if __name__ == "__main__":
    # instatiate image ETL class
    ingest = ImageIngestion()

    # read in and pre process training and testing data
    train_imgs_arr, train_labels_arr = ingest.image_pre_process(
        r"Labels//CSV Format//train_labels.csv"
    )
    test_imgs_arr, test_labels_arr = ingest.image_pre_process(
        r"Labels//CSV Format//test_labels.csv"
    )

    # instatiate model
    binary_classifier = BinaryClassifier(
        train_imgs_arr, train_labels_arr, test_imgs_arr, test_labels_arr
    )

    # save current model hyperparameters / architecture / etc
    binary_classifier.documentation()
    # build and compile binary classification model
    binary_classifier.create_and_compile_model(2, 20, (3, 3), (2, 2), 0.2)
    # train model
    binary_classifier.run_model(20)

    # instantiate predictor class
    predictions = MakePredictions(
        binary_classifier.history,
        test_imgs_arr,
        test_labels_arr,
        binary_classifier.model,
    )

    # generate predictions on test set
    predictions.make_predictions()
    # plot predictions and accuracy
    predictions.plot_predictions()
