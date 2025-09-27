import yaml
import click

import torch

from scrape_fa.paired_image_dataset import PairedImageDataset
from scrape_fa.simple_cnn import SimpleCNN

def prediction(dataset, model, target_dir, device):

    output_combined = None
    with torch.no_grad():
        output_combined = []
        for entry in dataset:
            feats = torch.stack((entry["feature"], entry["prewar"]))
            feats = feats.to(device)
            outputs = model(feats)
            output_combined.append(outputs.cpu().squeeze())

        PairedImageDataset.from_predictions(dataset, output_combined, target_dir)


@click.command()
@click.argument('config', type=click.Path(exists=True))
def cli(config) -> None:
    with open(config, 'r') as f:
        params = yaml.safe_load(f)
    required = ['prediction']
    for k in required:
        if k not in params:
            raise click.ClickException(f"Missing required config key: {k}")

    ds = PairedImageDataset(params['prediction']['input'])
    model = SimpleCNN.from_pth(params['prediction']['model'], model_args={"n_channels": 1, "n_classes": 1})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    prediction(ds, model, params['prediction']['output'], device)

if __name__ == '__main__':
    cli()