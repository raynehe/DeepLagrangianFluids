import torch
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import numpy as np
from pytorch3d.ops import knn_points


class MyParticleNetwork(torch.nn.Module):

    def __init__(
            self,
            kernel_size=[4, 4, 4],
            radius_scale=1.5,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,
            timestep=1 / 50,
            gravity=(0, -9.81, 0),
            other_feats_channels=0,
            ball_query_neighbor=0,
    ):
        super().__init__()
        self.layer_channels = [32, 64, 64, 3]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        self.ball_query_radius = np.float32(self.radius_scale * 3 *
                                        self.particle_radius)
        self.ball_query_neighbor = ball_query_neighbor
        self.timestep = timestep
        gravity = torch.FloatTensor(gravity)
        self.register_buffer('gravity', gravity)

        self._all_convs = []

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv

            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = conv_fn(kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)

            self._all_convs.append((name, conv))
            return conv

        self.conv0_fluid = Conv(name="conv0_fluid",
                                in_channels=4 + other_feats_channels,
                                filters=self.layer_channels[0],
                                activation=None)
        self.conv0_obstacle = Conv(name="conv0_obstacle",
                                   in_channels=3,
                                   filters=self.layer_channels[0],
                                   activation=None)
        self.dense0_fluid = torch.nn.Linear(in_features=4 +
                                            other_feats_channels,
                                            out_features=self.layer_channels[0])
        torch.nn.init.xavier_uniform_(self.dense0_fluid.weight)
        torch.nn.init.zeros_(self.dense0_fluid.bias)

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                in_ch *= 3
            out_ch = self.layer_channels[i]
            dense = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense.weight)
            torch.nn.init.zeros_(dense.bias)
            setattr(self, 'dense{0}'.format(i), dense)
            conv = Conv(name='conv{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None)
            setattr(self, 'conv{0}'.format(i), conv)
            self.denses.append(dense)
            self.convs.append(conv)

    def integrate_pos_vel(self, pos1, vel1):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity
        pos2 = pos1 + dt * (vel2 + vel1) / 2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """Apply the correction
        pos1,vel1 are the positions and velocities from the previous timestep
        pos2,vel2 are the positions after applying gravity and the integration step
        """
        dt = self.timestep
        pos = pos2 + pos_correction
        vel = (pos - pos1) / dt
        return pos, vel

    def compute_correction(self,
                           pos,
                           vel,
                           other_feats,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None):
        """Expects that the pos and vel has already been updated with gravity and velocity"""

        # compute the extent of the filters (the diameter)
        filter_extent = torch.tensor(self.filter_extent)

        # fluid_feats就三个特征，一个是全1，一个是速度，一个是其他特征
        fluid_feats = [torch.ones_like(pos[:, 0:1]), vel]
        if not other_feats is None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, axis=-1)

        self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos,
                                                filter_extent)
        # dense 就是全连接层
        self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
        self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box, pos,
                                                      filter_extent)

        feats = torch.cat([
            self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid
        ],
                          axis=-1)

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            inp_feats = F.relu(self.ans_convs[-1])
            ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            torch.ones_like(self.conv0_fluid.nns.neighbors_index,
                            dtype=torch.float32),
            self.conv0_fluid.nns.neighbors_row_splits)

        self.last_features = self.ans_convs[-2]

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * self.ans_convs[-1]
        return self.pos_correction

    def forward(self, inputs, fixed_radius_search_hash_table=None):
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        pos, vel, feats, box, box_feats = inputs

        ### Version1
        # feats is None here, but we need to change it to the surface attribute 表面/内部
        feats = self.compute_location(pos) # # (14664, 1)

        ### Version2
        # Send the points that are **ON** the surface to another branch of the network
        # This NEW branch is to be constructed.

        pos2, vel2 = self.integrate_pos_vel(pos, vel)
        pos_correction = self.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table)
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)

        return pos2_corrected, vel2_corrected

    # def init(self, feats_shape=None):
    # """Runs the network with dummy data to initialize the shape of all variables"""
    # pos = np.zeros(shape=(1, 3), dtype=np.float32)
    # vel = np.zeros(shape=(1, 3), dtype=np.float32)
    # if feats_shape is None:
    # feats = None
    # else:
    # feats = np.zeros(shape=feats_shape, dtype=np.float32)
    # box = np.zeros(shape=(1, 3), dtype=np.float32)
    # box_feats = np.zeros(shape=(1, 3), dtype=np.float32)

    # _ = self.__call__((pos, vel, feats, box, box_feats))

    def compute_location(self, pos):
        ''' 
        return: surface / inner attribute (12610, 1)
                assign 1 to the surface particles and 0 to the inner particles
        '''
        pos = pos.unsqueeze(0)
        self.ball_query_neighbor = int(pos.shape[1] * 0.04)
        dists = knn_points(p1=pos, p2=pos, K=self.ball_query_neighbor)[0] # (1,N,32)
        # ball_query_neighbor越大 表面(红色)越多
        dists_max = dists.squeeze(0)[:,-1].unsqueeze(1) # (N, 1)
        zeros = torch.zeros([dists_max.shape[0],1],dtype=torch.float).cuda() # (N, 1)
        ones = torch.ones([dists_max.shape[0],1],dtype=torch.float).cuda() # (N, 1)

        location = torch.where(dists_max > self.ball_query_radius, ones, zeros) 
        
        # print('ball_query_radius:',self.ball_query_radius)
        # print(dists_max[:10])
        # print(location[:10])
        self.plot_particle(pos, location)
        return location
    
    def plot_particle(self, pos, location):
        # import matplotlib
        # matplotlib.use('Agg')
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        pos = pos.squeeze(0) # (N, 3)
        location = location.squeeze(1) # (N,)
        surface = pos[location == 1][:]
        inner = pos[location == 0][:]
        surface = surface.cpu().detach().numpy()
        inner = inner.cpu().detach().numpy()
        print('surface:',surface.shape[0]) # surface - red
        print('inner:',inner.shape[0]) # inner - blue

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(surface[:,0], surface[:,1], surface[:,2], s=1, c='r', marker='.', alpha=0.5)
        # plt.show()
        ax.scatter(inner[:,0], inner[:,1], inner[:,2], s=1, c='b', marker='.', alpha=0.5)
        ax.legend()
        plt.show()
        plt.close()