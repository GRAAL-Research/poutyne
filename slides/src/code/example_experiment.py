from poutyne import Experiment

# Instead of `task`, you can provide your own loss function and metrics.
expt = Experiment('my_directory', network,
                  task='classifier', optimizer='sgd')
expt.train(train_loader, valid_loader,
           epochs=epochs,
           callbacks=callbacks,
           seed=42)
expt.test(test_loader)
