import os
from options.test_options import opt
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

if __name__ == '__main__':
    custom_parameters = opt().parse()  
    custom_parameters.num_threads = 0  
    custom_parameters.batch_size = 1    
    custom_parameters.serial_batches = True  
    custom_parameters.no_flip = True   
    custom_parameters.display_id = -1   
    dataset = create_dataset(custom_parameters) 
    model = create_model(custom_parameters)     
    model.setup(custom_parameters)               

    if custom_parameters.use_wandb:
        wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=custom_parameters.name, config=custom_parameters) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    web_dir = os.path.join(custom_parameters.results_dir, custom_parameters.name, '{}_{}'.format(custom_parameters.phase, custom_parameters.epoch))  # define the website directory
    if custom_parameters.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, custom_parameters.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (custom_parameters.name, custom_parameters.phase, custom_parameters.epoch))
    if custom_parameters.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= custom_parameters.num_test:  # only apply our model to custom_parameters.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=custom_parameters.aspect_ratio, width=custom_parameters.display_winsize, use_wandb=custom_parameters.use_wandb)
    webpage.save()  # save the HTML
