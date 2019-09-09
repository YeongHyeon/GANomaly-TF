import os, inspect, time, math

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))+"/.."

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gaussian_sample(batch_size, z_dim, mean=0, sigma=1):

    return np.random.normal(loc=mean, scale=sigma, size=(batch_size, z_dim)).astype(np.float32)

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def latent_plot(latent, y, n, savename=""):

    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def training(sess, saver, neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    summary_writer = tf.compat.v1.summary.FileWriter(PACK_PATH+'/Checkpoint', sess.graph)

    make_dir(path="results")
    result_list = ["tr_latent", "tr_resotring", "tr_latent_walk"]
    for result_name in result_list: make_dir(path=os.path.join("results", result_name))

    start_time = time.time()
    iteration = 0

    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()

    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        x_restore, z_enc = sess.run([neuralnet.x_hat, neuralnet.z_enc], \
            feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})
        if(neuralnet.z_dim == 2):
            latent_plot(latent=z_enc, y=y_tr, n=dataset.num_class, \
                savename=os.path.join("results", "tr_latent", "%08d.png" %(epoch)))
        save_img(contents=[x_tr, x_restore, (x_tr-x_restore)**2], \
            names=["Input\n(x)", "Restoration\n(x to x-hat)", "Difference"], \
            savename=os.path.join("results", "tr_resotring", "%08d.png" %(epoch)))

        if(neuralnet.z_dim == 2):
            x_values = np.linspace(-3, 3, test_sq)
            y_values = np.linspace(-3, 3, test_sq)
            z_latents = None
            for y_loc, y_val in enumerate(y_values):
                for x_loc, x_val in enumerate(x_values):
                    z_latent = np.reshape(np.array([y_val, x_val]), (1, neuralnet.z_dim))
                    if(z_latents is None): z_latents = z_latent
                    else: z_latents = np.append(z_latents, z_latent, axis=0)
            x_samples = sess.run(neuralnet.x_sample, \
                feed_dict={neuralnet.z:z_latents, neuralnet.batch_size:z_latents.shape[0]})
            plt.imsave(os.path.join("results", "tr_latent_walk", "%08d.png" %(epoch)), dat2canvas(data=x_samples))

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size) # y_tr does not used in this prj.

            _, summaries = sess.run([neuralnet.optimizer, neuralnet.summaries], \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]}, \
                options=run_options, run_metadata=run_metadata)
            restore_loss, kld, loss = sess.run([neuralnet.mean_restore, neuralnet.mean_kld, neuralnet.loss], \
                feed_dict={neuralnet.x:x_tr, neuralnet.batch_size:x_tr.shape[0]})
            summary_writer.add_summary(summaries, iteration)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration)  Restore:%.3f, KLD:%.3f, Total:%.3f" \
            %(epoch, epochs, iteration, restore_loss, kld, loss))
        saver.save(sess, PACK_PATH+"/Checkpoint/model_checker")
        summary_writer.add_run_metadata(run_metadata, 'epoch-%d' % epoch)

def test(sess, saver, neuralnet, dataset, batch_size):

    if(os.path.exists(PACK_PATH+"/Checkpoint/model_checker.index")):
        print("\nRestoring parameters")
        saver.restore(sess, PACK_PATH+"/Checkpoint/model_checker")

    print("\nTest...")

    make_dir(path="test")
    result_list = ["inbound", "outbound"]
    for result_name in result_list: make_dir(path=os.path.join("test", result_name))

    loss_list = []
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.
        restore_loss = sess.run(neuralnet.mean_restore, \
            feed_dict={neuralnet.x:x_te, neuralnet.batch_size:x_te.shape[0]})
        if(y_te[0] == 1):
            loss_list.append(restore_loss)

        if(terminator): break

    loss_list = np.asarray(loss_list)
    loss_avg, loss_std = np.average(loss_list), np.std(loss_list)
    outbound = loss_avg + (loss_std * 3)
    print("Loss  avg: %.3f, std: %.3f" %(loss_avg, loss_std))
    print("Outlier boundary: %.3f" %(outbound))

    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0
    z_enc_tot, y_te_tot = None, None
    loss4box = [[], [], [], [], [], [], [], [], [], []]
    while(True):
        x_te, y_te, terminator = dataset.next_test(1) # y_te does not used in this prj.

        x_restore, z_enc, restore_loss = sess.run([neuralnet.x_hat, neuralnet.z_enc, neuralnet.mean_restore], \
            feed_dict={neuralnet.x:x_te, neuralnet.batch_size:x_te.shape[0]})

        loss4box[y_te[0]].append(restore_loss)

        if(z_enc_tot is None):
            z_enc_tot = z_enc
            y_te_tot = y_te
        else:
            z_enc_tot = np.append(z_enc_tot, z_enc, axis=0)
            y_te_tot = np.append(y_te_tot, y_te, axis=0)

        outcheck = restore_loss > outbound
        fcsv.write("%d, %.3f, %r\n" %(y_te, restore_loss, outcheck))

        [h, w, c] = x_te[0].shape
        canvas = np.ones((h, w*3, c), np.float32)
        canvas[:, :w, :] = x_te[0]
        canvas[:, w:w*2, :] = x_restore[0]
        canvas[:, w*2:, :] = (x_te[0]-x_restore[0])**2
        if(outcheck):
            plt.imsave(os.path.join("test", "outbound", "%08d-%08d.png" %(testnum, int(restore_loss))), gray2rgb(gray=canvas))
        else:
            plt.imsave(os.path.join("test", "inbound", "%08d-%08d.png" %(testnum, int(restore_loss))), gray2rgb(gray=canvas))

        testnum += 1

        if(terminator): break

    boxplot(contents=loss4box, savename="test-box.png")

    if(neuralnet.z_dim == 2):
        latent_plot(latent=z_enc_tot, y=y_te_tot, n=dataset.num_class, \
            savename=os.path.join("test-latent.png"))
