from poutyne import Model, ModelCheckpoint, CSVLogger

callbacks = [
    ModelCheckpoint('last_epoch.ckpt'),
    ModelCheckpoint('best_epoch.ckpt', save_best_only=True,
                    monitor='val_acc', mode='max'),
    CSVLogger('log.csv'),
]

model = Model(network, 'sgd', 'cross_entropy',
              batch_metrics=['accuracy'], epoch_metrics=['f1'])
model.to(device)

model.fit_generator(train_loader, valid_loader,
                    epochs=num_epochs, callbacks=callbacks)

test_loss, (test_acc, test_f1) = model.evaluate_generator(test_loader)
print(f'Test: Loss: {test_loss}, Accuracy: {test_acc}, F1: {test_f1}')
