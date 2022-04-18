from pathlib import Path
import sys


# Package management BS
src_dir = Path(__file__).parent
module_dirs = ['datasets', 'models', 'training']
module_dirs = [str(Path(src_dir, module)) for module in module_dirs]

for module in module_dirs:
    sys.path.insert(1, module)


from training.train import train


def lots_of_training(dataset='tools', epochs=20): # Consider changing the default epochs to 20 in train.py

    # Baseline results
    train(  DATASET=dataset,
            DATA_TYPE='rating',
            NUM_EPOCHS=epochs)

    train(  DATASET=dataset,
            DATA_TYPE='helpfulness',
            NUM_EPOCHS=epochs)

    # Do more tokens result in higher accuracy?
    train(  DATASET=dataset,
            DATA_TYPE='rating',
            NUM_BERT_TOKENS=128,
            NUM_EPOCHS=epochs)

    train(  DATASET=dataset,
            DATA_TYPE='helpfulness',
            NUM_BERT_TOKENS=128,
            NUM_EPOCHS=epochs)

    
    # Is dropout actually bad?
    train(  DATASET=dataset,
            DATA_TYPE='rating',
            DROPOUT=0,
            NUM_EPOCHS=epochs)

    train(  DATASET=dataset,
            DATA_TYPE='helpfulness',
            DROPOUT=0,
            NUM_EPOCHS=epochs)


if __name__ == "__main__":


    # PLEASE READ THIS COMMET!!!!!!!!
    # If you want to change the hyperparameters found in train.py,
    # you can do so by adding them as all-caps keyword arguments to the
    # train() function
    # If you want to continue a training session from before,
    # pass the filename of the saved model (without the '.pt' extension)
    # as the first argument, called session_name
    train()
