import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import os
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    opt1 = TrainOptions().parse()
    opt1.isTrain = False
    opt1.model = 'pix2pix'
    opt1.which_model_netG = opt1.generator
    opt1.return_feature = True
    opt1.nThreads = 1   # test code only supports nThreads = 1
    opt1.batchSize = 1  # test code only supports batchSize = 1
    opt1.serial_batches = True  # no shuffle
    opt1.no_flip = True  # no flip
    opt1.display_id = -1  # no visdom display
    opt1.which_epoch = 'latest'

    # opt1.ntest = float('inf')
    # opt1.results_dir = './results/'
    # opt1.aspect_ratio = 1.0
    # opt1.phase = 'test'
    # opt1.which_epoch = 'latest'
    # opt1.how_many = 100000

    unet = create_model(opt1)
    unet.setup(opt1)
    features = dict()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            data1 = dict()
            data1['B'] = data['B']
            data1['A'] = data['A1']
            data1['A_paths'] = data['A_paths']
            data1['B_paths'] = data['B_paths']

            data1['A'] = data['A1']
            data1['A_paths'] = data['A1_paths']
            A_path = data1['A_paths'][0]
            if A_path in features:
                x1 = features[A_path]
            else:
                unet.set_input(data1)
                unet.test()
                x1 = unet.get_output()
                features[A_path] = x1

            data1['A'] = data['A']
            data1['A_paths'] = data['A_paths']
            A_path = data1['A_paths'][0]
            if A_path in features:
                x = features[A_path]
            else:
                unet.set_input(data1)
                unet.test()
                x = unet.get_output()
                features[A_path] = x

            data1['A'] = data['A3']
            data1['A_paths'] = data['A3_paths']
            A_path = data1['A_paths'][0]
            if A_path in features:
                x3 = features[A_path]
            else:
                unet.set_input(data1)
                unet.test()
                x3 = unet.get_output()
                features[A_path] = x3

            out = dict()
            out['x1'] = x1
            out['x'] = x
            out['x3'] = x3
            out['y'] = data['B']
            out['A1'] = data['A1']
            out['A'] = data['A']
            out['A3'] = data['A3']
            out['A_paths'] = data['A_paths']
            out['B_paths'] = data['B_paths']

            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(out)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest_lstm')
                model.save_networks(epoch)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest_lstm')
            model.save_networks(str(epoch) + '_lstm')

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
