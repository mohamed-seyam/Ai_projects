import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name = "f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)

        self.tp = tf.Variable(0, dtype=tf.int32)
        self.fp = tf.Variable(0, dtype=tf.int32)
        self.fn = tf.Variable(0, dtype=tf.int32)
        self.tn = tf.Variable(0, dtype=tf.int32)
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=2)
        self.tp.assign_add(conf_matrix[1][1])
        self.fp.assign_add(conf_matrix[0][1])
        self.fn.assign_add(conf_matrix[1][0])
        self.tn.assign_add(conf_matrix[0][0])
    
    def result(self):
        
        if (self.tp + self.fp == 0):
            precision = 1.0
        else:
            precision = self.tp / (self.tp + self.fp)
      
        if (self.tp + self.fn == 0):
            recall = 1.0
        else:
            recall = self.tp / (self.tp + self.fn)

        return tf.math.divide_no_nan(2 * precision * recall, precision + recall)
    
    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
        self.tn.assign(0)


if __name__ == "__main__":
    test_F1Score = F1Score()

    test_F1Score.tp = tf.Variable(2, dtype = 'int32')
    test_F1Score.fp = tf.Variable(5, dtype = 'int32')
    test_F1Score.tn = tf.Variable(7, dtype = 'int32')
    test_F1Score.fn = tf.Variable(9, dtype = 'int32')
    test_F1Score.result()
