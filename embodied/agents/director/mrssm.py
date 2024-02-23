import tensorflow as tf
from . import tfutils
from tensorflow.keras import mixed_precision as prec
import numpy as np



class MRSSM(tfutils.Module):
    """
    Hierarhical RSSM.
    """
    # TODO: add proper description


    def __init__(self):
        super().__init__()



class Block(tfutils.Module):
    """
    Stateful cell corresponding to one level of the hierarchy.
    """
    #TODO: add proper description

    def __init__(self, deter: int = 1024, stoch: int = 32, top: int = 32, down: int = 32, classes: int = 1):
        super().__init__()

        self.deter = deter
        self.stoch = stoch
        self.top = top
        self.down = down
        self._classes = classes
        self.h = None
        self.z = None
        self.s = None
        self.u = None
        self.e = None

        self.parent: Block = None
        self.child: Block = None

        self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

    def initial(self, batch_size):
        # zero initialization, equivalent to 'zeros' initialization in the original RSSM
        dtype = prec.global_policy().compute_dtype
        self.h = tf.zeros([batch_size, self.deter], dtype)
        self.z = tf.zeros([batch_size, self.stoch], dtype)
        self.top = tf.zeros([batch_size, self.top], dtype)
        self.down = tf.zeros([batch_size, self.down], dtype)

    def initial_imagine(self, batch_size):
        # zero initialization specific for imagination part
        block = Block(self.deter, self.stoch, self.top, self.down)
        block.initial(batch_size)
        return block # return a zero block that acts as the last state


    def imagine(self, action, states=None, training=False):
        # states corresponds to all the cells including the current one and all the cells upwards
        # when imagine, you do not want to update the stateful cell, the stateful cell should only be updated when observing
        # in planning, states are the stateful cells. 
        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if states[0] is None:
            states[0] = self.initial_imagine(action.shape[0]) # initialize the state if this is the beginning of the episode
        assert isinstance(states[0], dict), states[0]
        action = swap(action)
        prior = tfutils.scan(self.img_step, action, states[0], self._unroll)
        prior = {k: swap(v) for k, v in prior.items()}
        # if you go up the hierarchy
        if True: # TODO: add proper check
            priors = self.child.imagine(action, states[1:]) # priors is a list that contains the priors of all the above cells
            priors.append(prior)
        else:
            priors = [prior]
        return priors



    def observe(self, embed, action, is_first, training=False):
        # the original rssm observe function takes also the previous state as a paramater
        # this is not the case here since the state is already included in the cell itself.

        swap = lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape))))
        if self.h == None: # associate this check with a None previous state
            self.initial(action.shape[0]) # stateful initialization of the parameters
        step = lambda prev, inputs: self.obs_step(prev[0], *inputs)
        inputs = swap(action), swap(embed), swap(is_first)
        start = state, state
        post, prior = tfutils.scan(step, inputs, start, self._unroll)
        post = {k: swap(v) for k, v in post.items()}
        prior = {k: swap(v) for k, v in prior.items()}
        # if you go up the hierarchy
        if True: # TODO: add proper check
            priors, posteriors = self.child.observe(embed, action, is_first, training=False)
            priors.append(prior) # priors is a list that contains the priors of all the above cells
            posteriors.append(post) # posteriors is a list that contains the posteriors of all the above cells
        else:
            priors = [prior]
            posteriors = [post]
        return posteriors, priors


    def img_step(self, prev_state, prev_action):
        prev_stoch = self._cast(prev_state.z) # cast the previous stochastic state of the RSSM
        prev_action = self._cast(prev_action)
        if self._classes: # NOTE: not sure what classes refer to
            shape = prev_stoch.shape[:-2] + [self.stoch * self._classes]
            prev_stoch = tf.reshape(prev_stoch, shape)
        if len(prev_action.shape) > len(prev_stoch.shape):  # 2D actions.
            shape = prev_action.shape[:-2] + [np.prod(prev_action.shape[-2:])]
            prev_action = prev_action.reshape(shape)
        # calculate the determinstic part of the RSSM
        x = tf.concat([prev_stoch, prev_action], -1)
        x = self.get('img_in', Dense, **self._kw)(x)
        x, deter = self._gru(x, prev_state['deter']) # x and deter are the same thing here
        # concatenate deterministic part with the top down model
        x = tf.concat([x, prev_state.up], -1)
        # compute statistics for sampling
        x = self.get('img_out', Dense, **self._kw)(x)
        stats = self._stats_layer('img_stats', x)
        # determine distribution for sampling
        dist = self.get_dist(stats)
        # sample stochastic prior
        stoch = self._cast(dist.sample())
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior
    
    def _gru(self, x, deter): # GRU used for the deterministic part of the RSSM
        x = tf.concat([deter, x], -1)
        kw = {**self._kw, 'act': 'none', 'units': 3 * self._deter}
        x = self.get('gru', Dense, **kw)(x)
        reset, cand, update = tf.split(x, 3, -1)
        reset = tf.nn.sigmoid(reset)
        cand = tf.math.tanh(reset * cand)
        update = tf.nn.sigmoid(update - 1)
        deter = update * cand + (1 - update) * deter
        return deter, deter

    def obs_step(self, embed, is_first):
       prev_state, prev_action, is_first = tf.nest.map_structure(
            self._cast, (prev_state, prev_action, is_first)) 
        pass

    
    def hierarhical(self, x):
        pass

    def temporal(self, x):
        pass

    def _bottom_up(self, x):
        pass

    def _event_prior(self, x):
        pass

    def _stochastic(self, x):
        pass

    def _deter(self, x):
        pass

    def _temporal_stochastic(self, x):
        pass

    def _top_down(self, x):
        pass

class Dense(tfutils.Module):

  def __init__(self, units, act='none', norm='none', bias=True):
    self._units = units
    self._act = get_act(act)
    self._norm = norm
    self._bias = bias and norm == 'none'

  def __call__(self, x):
    kw = {}
    kw['use_bias'] = self._bias
    x = self.get('linear', tfkl.Dense, self._units, **kw)(x)
    x = self.get('norm', Norm, self._norm)(x)
    x = self._act(x)
    return x