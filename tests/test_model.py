
from transprosit import model
from transprosit import datamodules

model = model.PepTransformerModel()
datamodule = datamodules.PeptideDataModule(
    batch_size=2,
    base_dir="/home/jspaezp/git/ppptr/tests/data")
datamodule.setup()

for x in datamodule.val_dataloader():
    break

print([y.shape for y in x])

out = model(x[0], x[1], debug = True)
print([y.shape for y in out])
