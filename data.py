import pykitti
import numpy as np
import config
import gc


class StatefulDataGen(object):

    # Note some frames at the end of the sequence, and the
    # last sequence might be omitted to fit the examples
    # of timesteps x batch size8
    def __init__(self, base_dir, sequences):
        self.truncated_seq_sizes = []

        total_num_examples = 0

        for seq in sequences:
            seq_data = pykitti.odometry(base_dir, seq, frames=range(0, 20))
            num_frames = len(seq_data.poses)

            # less than timesteps number of frames will be discarded
            num_examples = (num_frames - 1) // config.timesteps
            self.truncated_seq_sizes.append(num_examples * config.timesteps)
            total_num_examples += num_examples

        # less than batch size number of examples will be discarded
        examples_per_batch = total_num_examples // config.batch_size
        # +1 adjusts for the extra image in the last time step
        frames_per_batch = examples_per_batch * (config.timesteps + 1)

        # since some examples will be discarded, readjust the truncated_seq_sizes
        deleted_frames = (total_num_examples - examples_per_batch * config.batch_size) * config.timesteps
        for i in range(len(self.truncated_seq_sizes) - 1, -1, -1):
            if self.truncated_seq_sizes[i] > deleted_frames:
                self.truncated_seq_sizes[i] -= deleted_frames
                break
            else:
                self.truncated_seq_sizes[i] = 0
                deleted_frames -= self.truncated_seq_sizes[i]

        # full = False
        # actual_used_examples = examples_per_batch * config.batch_size
        # for i in range(len(self.truncated_seq_sizes)):
        #     used_examples = self.truncated_seq_sizes[i] // config.timesteps
        #     actual_used_examples -= used_examples
        #
        #     if full:
        #         self.truncated_seq_sizes[i] = 0
        #
        #     if actual_used_examples < 0 and full == False:
        #         self.truncated_seq_sizes[i] += actual_used_examples * config.timesteps
        #         full = True

        # for storing all training
        self.input_frames = np.zeros(
            [frames_per_batch, config.batch_size, config.input_channels, config.input_height, config.input_width])
        self.poses = np.zeros([frames_per_batch, config.batch_size, 4, 4])

        num_image_loaded = 0
        for i_seq, seq in enumerate(sequences):
            print("Loading sequence " + seq)
            seq_data = pykitti.odometry(base_dir, seq)
            length = self.truncated_seq_sizes[i_seq]

            for i_img in range(length):

                if i_img % 50 == 0:
                    print("%f%% " % ((i_img / length) * 100), )

                i = num_image_loaded % frames_per_batch
                j = num_image_loaded // frames_per_batch

                # swap axis to channels first
                img = seq_data.get_cam0(i_img)
                img = img.resize((config.input_width, config.input_height))
                img = np.array(img)
                img = np.reshape(img, [img.shape[0], img.shape[1], config.input_channels])
                img = np.moveaxis(np.array(img), 2, 0)
                pose = seq_data.poses[i_img]

                self.input_frames[i, j] = img
                self.poses[i, j] = pose
                num_image_loaded += 1

                print(i_img)
                if i_img != 0 and i_img % config.timesteps == 0:

                    i = num_image_loaded % frames_per_batch
                    j = num_image_loaded // frames_per_batch
                    self.input_frames[i, j] = img
                    self.poses[i, j] = pose

                    num_image_loaded += 1
                    print(i_img)

                gc.collect()  # force garbage collection

            # make sure everything is fully loaded
        assert(num_image_loaded == frames_per_batch * config.batch_size)

    def next_batch(self):
        pass

    def config_next_epoch(self):
        pass
