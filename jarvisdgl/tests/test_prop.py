"""Training script test suite."""
from jarvisdgl.main import train_property_model
from jarvisdgl.train import train_dgl


def test_prop():
    """Test full training run with small batch size."""
    train_property_model(epochs=2, maxrows=16, batch_size=8)


def test_cgcnn_ignite():
    """Test CGCNN end to end training."""
    config = dict(
        target="formation_energy_peratom",
        epochs=2,
        n_train=16,
        n_val=16,
        batch_size=8,
    )
    result = train_dgl(config)
    print(result)
