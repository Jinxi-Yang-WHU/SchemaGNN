import os
import sys
import datetime

def create_logger(output_dir, model_name, dataset_name, task_name, hyperparams):
    """
    Creates a logger that writes to a file within a nested directory structure.

    The directory structure is: output_dir / model_name / hyperparameters / timestamp.log

    Args:
        output_dir: The base output directory.
        model_name: The name of the model.
        hyperparams: A dictionary of hyperparameters.

    Returns:
        A logger object.
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparam_str = "_".join(f"{k}={v}" for k, v in hyperparams.items())

    # Create nested directory structure
    log_dir = os.path.join(output_dir, model_name, dataset_name, task_name, hyperparam_str)
    print(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    filename = f"{timestamp}.log"
    filepath = os.path.join(log_dir, filename)

    logfile = open(filepath, 'w')

    class Logger:
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            #this flush method is needed for python 3 compatibility.
            #this handles the flush command by doing nothing.
            #you might want to specify some extra behavior here.
            pass

    sys.stdout = Logger(logfile)

    print(f"Logging to file: {filepath}")
    print(f"Model: {model_name}")
    print(f"Hyperparameters: {hyperparams}")

    return sys.stdout


# Example Usage
if __name__ == "__main__":
    hyperparams = {
        "num_bases": 1,
        "w1": 2,
        "w2": 3,
        "w3": 4
    }

    model_name = "Schema_GNN"
    dataset_name = "rel-f1"
    task_name = "driver-position"
    logger = create_logger("output", model_name, dataset_name, task_name, hyperparams)

    print("This is a test log message.")

    logger.log.close()
    sys.stdout = sys.stdout.terminal





        