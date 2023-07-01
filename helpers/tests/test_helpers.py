import unittest
import numpy as np 
import pandas as pd
from keras import backend as K

from helpers.df_ops import check_for_leakage
from helpers.df_ops import remove_data_leakage
from helpers.ai.loss import compute_class_freqs, get_weighted_loss

class TestLeakage(unittest.TestCase):
    # initialize the setup
    def setUp(self):
        self.df1 = pd.DataFrame({'patient_id': [0, 1, 2]})
        self.df2 = pd.DataFrame({'patient_id': [2, 3, 4]})
        
        self.df1_after_remvoing_leakage = pd.DataFrame({'patient_id': [0, 1]})
        self.df2_after_remvoing_leakage = pd.DataFrame({'patient_id': [3, 4]})

        self.df3 = pd.DataFrame({'patient_id': [0, 1, 2]})
        self.df4 = pd.DataFrame({'patient_id': [3, 4, 5]})

        self.df3_after_remvoing_leakage = pd.DataFrame({'patient_id': [0, 1, 2]})
        self.df4_after_remvoing_leakage = pd.DataFrame({'patient_id': [3, 4, 5]})

        


    
    def test_remove_data_leakage(self):

        self.assertEqual(remove_data_leakage(self.df1, self.df2, 'patient_id'), (self.df1, self.df2))
        self.assertEqual(remove_data_leakage(self.df3, self.df4, 'patient_id'), (self.df3, self.df4))

    def test_check_for_leakage(self):
        expected_output_1 = True
        expected_output_2 = False
        self.assertEqual(check_for_leakage(self.df1, self.df2, 'patient_id'), expected_output_1)
        self.assertEqual(check_for_leakage(self.df3, self.df4, 'patient_id'), expected_output_2)

class TestComputeClassFreqs(unittest.TestCase):
    def setUp(self):
        self.labels_matrix = np.array(
            [[1, 0, 0],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1],
             [1, 0, 1]]
        )
        self.expected_output = (np.array([0.8, 0.4, 0.8]), np.array([0.2, 0.6, 0.2]))
        
    def test_compute_class_freqs(self):
        output_freq_post, output_freq_neg = compute_class_freqs(self.labels_matrix)
        expected_freq_pos, expected_freq_neg = self.expected_output
        self.assertTrue(np.allclose(output_freq_post, expected_freq_pos, atol=1e-2))
        self.assertTrue(np.allclose(output_freq_neg, expected_freq_neg, atol=1e-2))


class TestGetWeightedLoss(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array(
            [[1, 1, 1],
             [1, 1, 0],
             [0, 1, 0],
             [1, 0, 1]]
        )

        

        self.y_pred_1 = 0.7*np.ones(self.y_true.shape)
        self.y_pred_2 = 0.3*np.ones(self.y_true.shape)

        self.expected_output_1 = np.float32(-0.4956203)
        self.expected_output_2 = np.float32(-0.4956203)
    def get_weighted_loss_test_case(self, sess):
        with sess.as_default() as sess:
            y_true = K.constant(np.array(
                [[1, 1, 1],
                [1, 1, 0],
                [0, 1, 0],
                [1, 0, 1]]
            ))
            
            w_p = np.array([0.25, 0.25, 0.5])
            w_n = np.array([0.75, 0.75, 0.5])
            
            y_pred_1 = K.constant(0.7*np.ones(y_true.shape))
            y_pred_2 = K.constant(0.3*np.ones(y_true.shape))
        
        return y_true.numpy(), w_p, w_n, y_pred_1.numpy(), y_pred_2.numpy()
    
    def test_get_weighted_loss(self):
        epsilon = 1
        sess = K.get_session()

        y_true, w_p, w_n, y_pred_1, y_pred_2 =  self.get_weighted_loss_test_case(sess)
       
        L = get_weighted_loss(w_p, w_n, epsilon)
        L1 = L(y_true, y_pred_1).numpy()
        L2 = L(y_true, y_pred_2).numpy()
        
        self.assertTrue(np.allclose(L1, self.expected_output_1, atol=1e-2))
        self.assertTrue(np.allclose(L2, self.expected_output_2, atol=1e-2))




if __name__ == "__main__":
    unittest.main(verbosity=2)