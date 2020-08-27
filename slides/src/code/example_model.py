from poutyne import Model

model = Model(network, optimizer, loss_function)
model.to(device)

model.fit_generator(train_loader, valid_loader,
                    epochs=num_epochs, callbacks=callbacks)

test_loss = model.evaluate_generator(test_loader)
