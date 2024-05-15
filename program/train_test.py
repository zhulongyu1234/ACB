import torch.optim as optim
import time
import torch
import torch.nn.functional as F
import os


def train(args, epoch, train_loader, len_src_dataset, device, model, **kwargs):
    # learning_rate = args.lr / math.pow((1 + 10 * (epoch - 1) / num_epoch), 0.75)
    learning_rate = args.lr
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    correct_sample_num = 0
    one_epoch_time = 0
    model.train()

    for i, (data, label) in enumerate(train_loader):
        start_time = time.time()
        data, label = data.to(device), label.to(device)
        label = label - 1

        optimizer.zero_grad()
        label_src_pred, _ = model(data)
        loss = F.nll_loss(F.log_softmax(label_src_pred, dim=1), label.long())
        loss.backward()
        optimizer.step()

        one_batch_time = time.time() - start_time
        one_epoch_time += one_batch_time

        pred = label_src_pred.data.max(1)[1]
        correct_sample_num += pred.eq(label.data.view_as(pred)).cpu().sum()
        if (i + 1) % args.log_interval == 0:
            logs1 = 'Train Epoch: {} [current samples:{}/{} ({:.0f}%)]\n'.format(epoch, i*args.batch_size+len(label), len_src_dataset,
                                                                 100. * (i + 1) / len(train_loader))
            logs2 = 'loss: {:.6f}\n'.format(loss.item())
            print(logs1, end='')
            print(logs2, end='')

    accuracy = correct_sample_num / len_src_dataset
    logs3 = '[epoch: {:4}]  Train Accuracy: {:.4f} | epoch time:{}\n'.format(epoch, accuracy, one_epoch_time)
    print(logs3, end='')

    return model, accuracy


def test(model, test_loader, device, task_logit_dir):
    model.eval()
    loss = 0
    correct_sample_num = 0
    pred_list, label_list = [], []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            label = label - 1

            label_predict, _ = model(data)

            pred = label_predict.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(label_predict, dim=1), label.long()).item()

            correct_sample_num += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss /= len(test_loader)
        accuracy = 100. * correct_sample_num / len(test_loader.dataset)
        logs = 'test loss of one batch: {:.4f}, test Accuracy: {}/{} ({:.2f}%)\n'.format(loss, correct_sample_num, len(
            test_loader.dataset), accuracy)
        print(logs, end='')
        with open(os.path.join(task_logit_dir, 'log_accuracy.txt'), 'a') as file:
            file.write(logs)
    return accuracy, pred_list, label_list


