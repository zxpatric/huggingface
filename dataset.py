from datasets import load_dataset_builder, load_dataset

def load_a_dataset_to_test():
    ds_builder = load_dataset_builder("cornell-movie-review-data/rotten_tomatoes")

    print(ds_builder.info.description, ds_builder.info.features)

def load_a_dataset():
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")

    print(dataset)

def split_a_dataset():
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
    # dataset["train"]

    split_dataset = dataset.train_test_split(test_size=0.1)

    # split twice above. one train, validation, test and the other train and test within the training.
    print(split_dataset)


    


if __name__ == "__main__":
    # load_a_dataset_to_test()
    load_a_dataset()
    split_a_dataset()
