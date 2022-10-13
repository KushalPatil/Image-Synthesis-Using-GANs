import time
from options.train_options import TrainOptions
from data import create_input
from models import create_model
from util.visualizer import Visualizer

if __name__ == '__main__':
    opt = TrainOptions().parse()   
    input = create_input(opt)  
    input_len = len(input)    
    print('The number of training images = %d' % input_len)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    loops_no = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  
        t = time.time()    
        epoch = 0                  
        visualizer.reset()              
        model.update_learning_rate()    
        for i, data in enumerate(input):  
            iter_start_time = time.time()  
            if loops_no % opt.print_freq == 0:
                t_data = iter_start_time - t

            loops_no += opt.batch_size
            epoch += opt.batch_size
            model.set_input(data)         
            model.optimize_parameters()  

            if loops_no % opt.display_freq == 0:  
                save_result = loops_no % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if loops_no % opt.print_freq == 0: 
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch) / input_len, losses)

            if loops_no % opt.save_latest_freq == 0:   
                print('saving the latest model (epoch %d, loops_no %d)' % (epoch, loops_no))
                save_suffix = 'iter_%d' % loops_no if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            t = time.time()
        if epoch % opt.save_epoch_freq == 0:              
            
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
