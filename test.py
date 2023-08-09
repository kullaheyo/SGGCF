import torch

def test(model, loader, device, CONFIG, metrics):
    '''
    test for dot-based model
    '''
    model.eval()
    for metric in metrics:
        metric.start()
    with torch.no_grad():
        rs = model.propagate() 
        for groups, ground_truth_g_i, train_mask_g_i in loader:
            pred_i = model.evaluate(rs, groups.to(device))
            pred_i -= 1e8*train_mask_g_i.to(device)
            for metric in metrics:
                metric(pred_i, ground_truth_g_i.to(device))
    for metric in metrics:
        metric.stop()
        print('{}:{}'.format(metric.get_title(), metric.metric), end='\t')
    print('')
    return metrics