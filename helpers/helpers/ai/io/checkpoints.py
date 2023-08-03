import os 
from helpers.text.sort.text_sort import sort_alphanumeric

def load_weights_from_checkpoints(model, checkpoint_path: str) -> int:
        """Given a folder of checkpoints, load the latest checkpoint into the model"""
        if model is None:
            print("ERROR: Model was not initialized or loaded from file")
            return 0

        if os.path.exists(checkpoint_path) == False:
            print("WARNING: Checkpoint directory not found")
            return 0

        checkpoint_files = [
            file for file in os.listdir(checkpoint_path) if file.endswith("." + "h5")
        ]

        max_epoch_number = 0

        if len(checkpoint_files) == 0:
            print(
                "WARNING: Checkpoint directory empty. Path provided: "
                + str(checkpoint_path)
            )
            return max_epoch_number

        checkpoint_files = sort_alphanumeric(checkpoint_files)
        print("------------------------------------------------------")
        print("Available checkpoint files: {}".format(checkpoint_files))
        max_epoch_number = int(re.findall(r"\d+", checkpoint_files[-1][:-3])[0])
        max_epoch_filename = checkpoint_files[-1]

        print("Latest epoch checkpoint file name: {}".format(max_epoch_filename))
        print("Resuming training from epoch: {}".format(int(max_epoch_number) + 1))
        print("------------------------------------------------------")
        model.load_weights(f"{checkpoint_path}/{max_epoch_filename}")
        return model, max_epoch_number
