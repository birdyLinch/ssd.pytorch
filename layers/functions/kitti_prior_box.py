import torch
from math import sqrt as sqrt
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.image_size_x = cfg['x_dim']###############
        self.image_size_y = cfg['y_dim']############
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps_x = cfg['feature_maps_x']#####################
        self.feature_maps_y = cfg['feature_maps_y']######################
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps_x = cfg['steps_x']######################
        self.steps_y = cfg['steps_y']#########################
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        if self.version == 'v2':
            for k, f in enumerate(self.feature_maps_x):##########
                ff=self.feature_maps_y[k]###############
                # for i, j in product(range(f), repeat=2):
                for i, j in product(range(f), range(ff)):##########
                    f_k_x = self.image_size_x / self.steps_x[k]###########
                    f_k_y = self.image_size_y / self.steps_y[k]##############
                    # unit center x,y
                    cx = (i + 0.5) / f_k_x###############
                    cy = (j + 0.5) / f_k_y##################
                

                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k_x = self.min_sizes[k]/self.image_size_x##################
                    s_k_y = self.min_sizes[k]/self.image_size_y##################
                    mean += [cx, cy, s_k_x, s_k_y]#############################

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    s_k_prime_x = sqrt(s_k_x * (self.max_sizes[k]/self.image_size_x))################
                    s_k_prime_y = sqrt(s_k_y * (self.max_sizes[k]/self.image_size_y))################
                    mean += [cx, cy, s_k_prime_x, s_k_prime_y]#########################

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:###########################
                        mean += [cx, cy, s_k_x*sqrt(ar), s_k_x/sqrt(ar)]####################
                        mean += [cx, cy, s_k_y/sqrt(ar), s_k_y*sqrt(ar)]###########################3
        else:
            # original version generation of prior (default) boxes
            # for i, k in enumerate(self.feature_maps):
            #     step_x = step_y = self.image_size/k
            #     for h, w in product(range(k), repeat=2):
            #         c_x = ((w+0.5) * step_x)
            #         c_y = ((h+0.5) * step_y)
            #         c_w = c_h = self.min_sizes[i] / 2
            #         s_k = self.image_size  # 300
            #         # aspect_ratio: 1,
            #         # size: min_size
            #         mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
            #                  (c_x+c_w)/s_k, (c_y+c_h)/s_k]
            #         if self.max_sizes[i] > 0:
            #             # aspect_ratio: 1
            #             # size: sqrt(min_size * max_size)/2
            #             c_w = c_h = sqrt(self.min_sizes[i] *
            #                              self.max_sizes[i])/2
            #             mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
            #                      (c_x+c_w)/s_k, (c_y+c_h)/s_k]
            #         # rest of prior boxes
            #         for ar in self.aspect_ratios[i]:
            #             if not (abs(ar-1) < 1e-6):
            #                 c_w = self.min_sizes[i] * sqrt(ar)/2
            #                 c_h = self.min_sizes[i] / sqrt(ar)/2
            #                 mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
            #                          (c_x+c_w)/s_k, (c_y+c_h)/s_k]
            print('Error use v1')
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
