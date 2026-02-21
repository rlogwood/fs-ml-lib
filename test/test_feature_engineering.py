"""
Pytest tests for feature_engineering module
"""

import pytest
import numpy as np
import pandas as pd
import feature_engineering as fe


@pytest.fixture
def sample_df():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame({
        'revol.util': np.random.uniform(0, 100, n_samples),
        'inq.last.6mths': np.random.randint(0, 10, n_samples),
        'installment': np.random.uniform(50, 1000, n_samples),
        'log.annual.inc': np.random.uniform(9, 13, n_samples),
        'days.with.cr.line': np.random.uniform(365, 7300, n_samples),
        'dti': np.random.uniform(0, 30, n_samples)
    })


class TestValidation:
    """Tests for validation functions"""

    def test_validate_features_for_engineering_valid(self, sample_df):
        """Test validation with valid dataframe"""
        is_valid, missing = fe.validate_features_for_engineering(sample_df)
        assert is_valid
        assert len(missing) == 0

    def test_validate_features_for_engineering_missing_columns(self, sample_df):
        """Test validation with missing columns"""
        df_missing = sample_df.drop(columns=['revol.util', 'dti'])
        is_valid, missing = fe.validate_features_for_engineering(df_missing)
        assert not is_valid
        assert len(missing) == 2
        assert 'revol.util' in missing
        assert 'dti' in missing


class TestFeatureCreation:
    """Tests for feature creation"""

    def test_create_loan_features_adds_correct_features(self, sample_df):
        """Test that all expected features are added"""
        df_enhanced = fe.create_loan_features(sample_df)

        original_cols = set(sample_df.columns)
        new_cols = set(df_enhanced.columns) - original_cols

        expected_features = {
            'credit_util_ratio',
            'annual_inquiry_rate',
            'debt_burden',
            'credit_history_years',
            'high_debt',
            'risk_score'
        }

        assert new_cols == expected_features

    def test_create_loan_features_preserves_original_data(self, sample_df):
        """Test that original data is preserved"""
        df_enhanced = fe.create_loan_features(sample_df)

        for col in sample_df.columns:
            pd.testing.assert_series_equal(
                sample_df[col],
                df_enhanced[col],
                check_names=True
            )

    def test_create_loan_features_preserves_shape(self, sample_df):
        """Test that number of rows is preserved"""
        df_enhanced = fe.create_loan_features(sample_df)
        assert len(df_enhanced) == len(sample_df)


class TestCreditUtilRatio:
    """Tests for credit utilization ratio"""

    def test_credit_util_ratio_calculation(self, sample_df):
        """Test credit utilization ratio is revol.util / 100"""
        df_enhanced = fe.create_loan_features(sample_df)

        expected = sample_df['revol.util'] / 100
        pd.testing.assert_series_equal(
            df_enhanced['credit_util_ratio'],
            expected,
            check_names=False
        )

    def test_credit_util_ratio_range(self, sample_df):
        """Test credit util ratio is in range 0-1"""
        df_enhanced = fe.create_loan_features(sample_df)

        assert (df_enhanced['credit_util_ratio'] >= 0).all()
        assert (df_enhanced['credit_util_ratio'] <= 1).all()


class TestAnnualInquiryRate:
    """Tests for annual inquiry rate"""

    def test_annual_inquiry_rate_calculation(self, sample_df):
        """Test annual inquiry rate is inq.last.6mths * 2"""
        df_enhanced = fe.create_loan_features(sample_df)

        expected = sample_df['inq.last.6mths'] * 2
        pd.testing.assert_series_equal(
            df_enhanced['annual_inquiry_rate'],
            expected,
            check_names=False
        )


class TestDebtBurden:
    """Tests for debt burden"""

    def test_debt_burden_calculation(self, sample_df):
        """Test debt burden is installment / exp(log.annual.inc)"""
        df_enhanced = fe.create_loan_features(sample_df)

        expected = sample_df['installment'] / np.exp(sample_df['log.annual.inc'])
        pd.testing.assert_series_equal(
            df_enhanced['debt_burden'],
            expected,
            check_names=False,
            atol=1e-6
        )

    def test_debt_burden_positive(self, sample_df):
        """Test debt burden values are positive"""
        df_enhanced = fe.create_loan_features(sample_df)
        assert (df_enhanced['debt_burden'] > 0).all()


class TestCreditHistoryYears:
    """Tests for credit history years"""

    def test_credit_history_years_calculation(self, sample_df):
        """Test credit history is days / 365.25"""
        df_enhanced = fe.create_loan_features(sample_df)

        expected = sample_df['days.with.cr.line'] / 365.25
        pd.testing.assert_series_equal(
            df_enhanced['credit_history_years'],
            expected,
            check_names=False,
            atol=1e-6
        )


class TestHighDebtFlag:
    """Tests for high debt binary flag"""

    def test_high_debt_flag_binary(self, sample_df):
        """Test high_debt is binary 0 or 1"""
        df_enhanced = fe.create_loan_features(sample_df)
        assert df_enhanced['high_debt'].isin([0, 1]).all()

    def test_high_debt_flag_logic(self, sample_df):
        """Test high_debt logic: dti > 20 OR revol.util > 80"""
        df_enhanced = fe.create_loan_features(sample_df)

        expected = ((sample_df['dti'] > 20) |
                   (sample_df['revol.util'] > 80)).astype(int)
        pd.testing.assert_series_equal(
            df_enhanced['high_debt'],
            expected,
            check_names=False
        )


class TestRiskScore:
    """Tests for risk score"""

    def test_risk_score_exists_and_numeric(self, sample_df):
        """Test risk_score exists and is numeric"""
        df_enhanced = fe.create_loan_features(sample_df)

        assert 'risk_score' in df_enhanced.columns
        assert pd.api.types.is_numeric_dtype(df_enhanced['risk_score'])

    def test_risk_score_reasonable_range(self, sample_df):
        """Test risk_score is in reasonable range"""
        df_enhanced = fe.create_loan_features(sample_df)

        assert (df_enhanced['risk_score'] >= 0).all()
        assert (df_enhanced['risk_score'] <= 2).all()


class TestUtilityFunctions:
    """Tests for utility functions"""

    def test_get_engineered_feature_names(self):
        """Test getting list of feature names"""
        feature_names = fe.get_engineered_feature_names()

        assert len(feature_names) == 6
        assert 'credit_util_ratio' in feature_names
        assert 'risk_score' in feature_names


class TestEdgeCases:
    """Tests for edge cases"""

    def test_feature_engineering_with_edge_values(self):
        """Test with edge case values"""
        df_edge = pd.DataFrame({
            'revol.util': [0, 100, 50],
            'inq.last.6mths': [0, 33, 5],
            'installment': [15.67, 940.14, 500],
            'log.annual.inc': [7.55, 14.53, 11],
            'days.with.cr.line': [178.96, 17639.96, 4000],
            'dti': [0, 29.96, 15]
        })

        df_enhanced = fe.create_loan_features(df_edge)

        # Check no NaN or inf values
        for col in fe.get_engineered_feature_names():
            assert not df_enhanced[col].isna().any(), f"NaN values found in {col}"
            assert not np.isinf(df_enhanced[col]).any(), f"Inf values found in {col}"


class TestPrepareDataForTraining:
    """Tests for prepare_data_for_training function"""

    def test_prepare_data_basic(self):
        """Test basic data preparation"""
        if not hasattr(fe, 'prepare_data_for_training'):
            pytest.skip("prepare_data_for_training not available")

        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
            'category': np.random.choice(['A', 'B', 'C'], n),
            'target': np.random.randint(0, 2, n)
        })

        result = fe.prepare_data_for_training(df, target_col='target')

        # Check that PreparedData object is returned
        assert hasattr(result, 'X_train')
        assert hasattr(result, 'y_train')
        assert hasattr(result, 'X_test')
        assert hasattr(result, 'y_test')
        assert hasattr(result, 'X_val')
        assert hasattr(result, 'y_val')

    def test_prepare_data_shapes(self):
        """Test that data shapes are correct"""
        if not hasattr(fe, 'prepare_data_for_training'):
            pytest.skip("prepare_data_for_training not available")

        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            'feature1': np.random.randn(n),
            'feature2': np.random.randn(n),
            'target': np.random.randint(0, 2, n)
        })

        result = fe.prepare_data_for_training(df, target_col='target')

        # Check total samples
        total = len(result.X_train) + len(result.X_val) + len(result.X_test)
        assert total == n

        # Check that training is larger than test
        assert len(result.X_train) > len(result.X_test)

    def test_prepare_data_categorical_encoding(self):
        """Test that categorical variables are encoded"""
        if not hasattr(fe, 'prepare_data_for_training'):
            pytest.skip("prepare_data_for_training not available")

        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            'numeric': np.random.randn(n),
            'category': np.random.choice(['A', 'B', 'C'], n),
            'target': np.random.randint(0, 2, n)
        })

        result = fe.prepare_data_for_training(df, target_col='target')

        # After encoding, should have more than 1 feature (numeric + encoded categories)
        assert result.X_train.shape[1] > 1

    def test_prepare_data_scaling(self):
        """Test that features are scaled"""
        if not hasattr(fe, 'prepare_data_for_training'):
            pytest.skip("prepare_data_for_training not available")

        np.random.seed(42)
        n = 100

        df = pd.DataFrame({
            'feature1': np.random.uniform(0, 1000, n),  # Large scale
            'feature2': np.random.uniform(0, 1, n),     # Small scale
            'target': np.random.randint(0, 2, n)
        })

        result = fe.prepare_data_for_training(df, target_col='target')

        # Check that scaled data has mean ~0 and std ~1
        mean = np.mean(result.X_train, axis=0)
        std = np.std(result.X_train, axis=0)

        assert np.allclose(mean, 0, atol=0.1)
        assert np.allclose(std, 1, atol=0.2)


class TestIntegration:
    """Integration tests"""

    def test_full_pipeline(self):
        """Test complete feature engineering pipeline"""
        np.random.seed(123)
        n = 50

        df = pd.DataFrame({
            'credit.policy': np.random.randint(0, 2, n),
            'purpose': np.random.choice(['debt_consolidation', 'credit_card'], n),
            'int.rate': np.random.uniform(0.06, 0.22, n),
            'installment': np.random.uniform(50, 900, n),
            'log.annual.inc': np.random.uniform(9, 13, n),
            'dti': np.random.uniform(0, 30, n),
            'days.with.cr.line': np.random.uniform(500, 7000, n),
            'revol.bal': np.random.randint(0, 100000, n),
            'revol.util': np.random.uniform(0, 100, n),
            'inq.last.6mths': np.random.randint(0, 15, n),
            'delinq.2yrs': np.random.randint(0, 5, n),
            'pub.rec': np.random.randint(0, 3, n),
            'not.fully.paid': np.random.randint(0, 2, n)
        })

        # Validate before engineering
        is_valid, missing = fe.validate_features_for_engineering(df)
        assert is_valid, f"Missing columns: {missing}"

        # Apply feature engineering
        df_enhanced = fe.create_loan_features(df)

        # Validate results
        assert len(df_enhanced) == n
        assert len(df_enhanced.columns) == len(df.columns) + 6

        # Check all engineered features exist
        for feature in fe.get_engineered_feature_names():
            assert feature in df_enhanced.columns

    def test_full_pipeline_with_preparation(self):
        """Test feature engineering followed by data preparation"""
        if not hasattr(fe, 'prepare_data_for_training'):
            pytest.skip("prepare_data_for_training not available")

        np.random.seed(123)
        n = 100

        df = pd.DataFrame({
            'credit.policy': np.random.randint(0, 2, n),
            'purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home'], n),
            'int.rate': np.random.uniform(0.06, 0.22, n),
            'installment': np.random.uniform(50, 900, n),
            'log.annual.inc': np.random.uniform(9, 13, n),
            'dti': np.random.uniform(0, 30, n),
            'days.with.cr.line': np.random.uniform(500, 7000, n),
            'revol.bal': np.random.randint(0, 100000, n),
            'revol.util': np.random.uniform(0, 100, n),
            'inq.last.6mths': np.random.randint(0, 15, n),
            'delinq.2yrs': np.random.randint(0, 5, n),
            'pub.rec': np.random.randint(0, 3, n),
            'not.fully.paid': np.random.randint(0, 2, n)
        })

        # Engineer features
        df_enhanced = fe.create_loan_features(df)

        # Prepare for training
        result = fe.prepare_data_for_training(df_enhanced, target_col='not.fully.paid')

        # Validate
        assert result.X_train is not None
        assert result.y_train is not None
        assert len(result.X_train) > 0
        assert len(result.y_train) > 0
