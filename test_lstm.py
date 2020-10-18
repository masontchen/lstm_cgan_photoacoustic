import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
import time


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)

    opt1 = TestOptions().parse()
    opt1.model = 'pix2pix'
    opt1.which_model_netG = opt1.generator
    opt1.return_feature = True
    opt1.nThreads = 1  # test code only supports nThreads = 1
    opt1.batchSize = 1  # test code only supports batchSize = 1
    opt1.serial_batches = True  # no shuffle
    opt1.no_flip = True  # no flip
    opt1.display_id = -1  # no visdom display
    opt1.which_epoch = 'latest'

    unet = create_model(opt1)
    unet.setup(opt1)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        data1 = dict()
        data1['B'] = data['B']
        data1['A'] = data['A1']
        data1['A_paths'] = data['A_paths']
        data1['B_paths'] = data['B_paths']

        unet.set_input(data1)
        unet.test()
        x1 = unet.get_output()
        data1['A'] = data['A']
        unet.set_input(data1)
        unet.test()
        x = unet.get_output()
        data1['A'] = data['A3']
        unet.set_input(data1)
        unet.test()
        x3 = unet.get_output()

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

        model.set_input(out)
        start_time = time.time()
        model.test()
        # print(time.time() - start_time)
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    webpage.save()
