import tensorflow as tf

class YoutubeLikeNetwork:
    def __init__(self, user_model_mode='DEFAULT'):
        
        # --- placeholders
        self._precomputed_item_vectors = tf.placeholder(shape=[None, 128], dtype=tf.float32)        
        self._profile_item_indexes = tf.placeholder(shape=[None], dtype=tf.int32)
        self._candidate_item_indexes = tf.placeholder(shape=[None], dtype=tf.int32)
            
        # ---- user profile vector
        
        # profile item vectors average
        tmp = tf.gather(self._precomputed_item_vectors, self._profile_item_indexes) 
        self._profile_items_average = tf.reshape(tf.reduce_mean(tmp, axis=0), (1, 128))
        
        if user_model_mode == 'BIGGER':
            # user hidden layer 1
            self._user_hidden_1 = tf.layers.dense(
                inputs=self._profile_items_average,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden_1'
            )
            # user hidden layer 2
            self._user_hidden_2 = tf.layers.dense(
                inputs=self._user_hidden_1,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden_2'
            )
            # user final vector
            self._user_vector = tf.layers.dense(
                inputs=self._user_hidden_2,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        elif user_model_mode == 'BIG':
            # user hidden layer
            self._user_hidden = tf.layers.dense(
                inputs=self._profile_items_average,
                units=256,
                activation=tf.nn.selu,
                name='user_hidden'
            )
            # user final vector
            self._user_vector = tf.layers.dense(
                inputs=self._user_hidden,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        elif user_model_mode == 'DEFAULT':
            # user hidden layer
            self._user_hidden = tf.layers.dense(
                inputs=self._profile_items_average,
                units=128,
                activation=tf.nn.selu,
                name='user_hidden'
            )
            # user final vector
            self._user_vector = tf.layers.dense(
                inputs=self._user_hidden,
                units=128,
                activation=tf.nn.selu,
                name='user_vector'
            )
        else: assert False
        
        # ---- candidate item vectors
        self._candidate_item_vectors = tf.gather(
            self._precomputed_item_vectors, self._candidate_item_indexes)
        
        # ---- match scores
        self._match_scores = tf.reduce_sum(
            tf.multiply(self._user_vector, self._candidate_item_vectors), 1)
    
    def get_match_scores(
            self, sess,
            precomputed_item_vectors,
            profile_item_indexes,
            candidate_item_indexes):
        return sess.run(self._match_scores, feed_dict={
            self._precomputed_item_vectors: precomputed_item_vectors,
            self._profile_item_indexes: profile_item_indexes,
            self._candidate_item_indexes: candidate_item_indexes,
        })