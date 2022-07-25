import pandas as pd
from pythae.pipelines import TrainingPipeline
from pythae.models import VAE, VAEConfig
from pythae.trainers import BaseTrainerConfig
from sklearn.model_selection import train_test_split

# Set up the training configuration
if __name__ == '__main__':
    data = pd.read_csv('Data/DCFCfinal.csv')
    y = data['Converted Cost'].to_numpy()
    del data['Converted Cost']
    x = data.to_numpy()
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=4)
    my_training_config = BaseTrainerConfig(
        output_dir='my_model',
        num_epochs=50,
        learning_rate=1e-3,
        batch_size=200,
        steps_saving=None
    )
    # Set up the model configuration
    my_vae_config = model_config = VAEConfig(
        input_dim=(1, 2, 5),
        latent_dim=10
    )
    # Build the model
    my_vae_model = VAE(
        model_config=my_vae_config
    )
    # Build the Pipeline
    pipeline = TrainingPipeline(
        training_config=my_training_config,
        model=my_vae_model
    )
    # Launch the Pipeline
    pipeline(
        train_data=x,
        eval_data=y
    )
