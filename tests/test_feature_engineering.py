import unittest
import numpy as np
import pandas as pd
from src.feature_engineering import engineer_features

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        train_data = {
            "DAYS_EMPLOYED": [-1000, 365243, -2500],
            "AMT_CREDIT": [100000.0, 50000.0, 75000.0],
            "AMT_INCOME_TOTAL": [50000.0, 25000.0, 30000.0],
            "AMT_ANNUITY": [10000.0, 5000.0, 7000.0],
            "AMT_GOODS_PRICE": [90000.0, 50000.0, 75000.0],
            "CNT_FAM_MEMBERS": [2, 1, 3],
            "DAYS_BIRTH": [-10000, -20000, -15000],
            "DAYS_REGISTRATION": [-1000, -2000, -1200],
            "DAYS_ID_PUBLISH": [-500, -1000, -800],
            "DAYS_LAST_PHONE_CHANGE": [-100, -200, -150],
            "OWN_CAR_AGE": [5.0, np.nan, 2.0],
            "EXT_SOURCE_1": [0.5, 0.6, 0.4],
            "EXT_SOURCE_2": [0.6, 0.7, 0.5],
            "EXT_SOURCE_3": [0.7, 0.8, 0.6],
            "ORGANIZATION_TYPE": ["Business Entity Type 3", "XNA", "Self-employed"],
            "NAME_EDUCATION_TYPE": ["Higher education", "Secondary / secondary special", "Higher education"],
            "OCCUPATION_TYPE": ["Laborers", "XNA", "Managers"],
            "TARGET": [0, 1, 0]
        }
        test_data = train_data.copy()
        test_data["TARGET"] = [np.nan, np.nan, np.nan]
        
        self.train_df = pd.DataFrame(train_data)
        self.test_df = pd.DataFrame(test_data)

    def test_pensioner_and_employed_cleaning(self):
        train_feat, _, _, _ = engineer_features(self.train_df, self.test_df)
        self.assertEqual(train_feat["IS_PENSIONER"].iloc[0], 0)
        self.assertEqual(train_feat["IS_PENSIONER"].iloc[1], 1)
        self.assertTrue(np.isnan(train_feat["DAYS_EMPLOYED_CLEAN"].iloc[1]))

    def test_financial_ratios_and_uncovered_credit(self):
        train_feat, _, _, _ = engineer_features(self.train_df, self.test_df)
        expected_c2i = 100000.0 / (50000.0 + 1e-5)
        self.assertAlmostEqual(train_feat["CREDIT_TO_INCOME"].iloc[0], expected_c2i, places=4)
        self.assertAlmostEqual(train_feat["UNCOVERED_CREDIT"].iloc[0], 10000.0, places=4)
        self.assertAlmostEqual(train_feat["UNCOVERED_CREDIT"].iloc[1], 0.0, places=4)
        self.assertIn("LOAN_TERM_CLUSTER", train_feat.columns)

if __name__ == "__main__":
    unittest.main()
