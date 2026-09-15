from sklearn.feature_selection import mutual_info_regression
import numpy as np
import statsmodels.api as sm
from tqdm.notebook import tqdm
import pandas as pd
from sklearn.decomposition import PCA

class Selector:
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def fit(self, X, y):
        '''
        Fits the selector to the data.
        '''
        pass
    
    def transform(self, X, y):
        '''
        Transforms the data.
        Returns the transformed data.
        '''
        pass
    
    def fit_transform(self, X, y):
        '''
        Fits the selector to the data and transforms the data.
        Returns the transformed data.
        '''
        if self.verbose:
            print('Selecting via', self.__class__.__name__)
        self.fit(X, y)
        if self.verbose:
            print('Transforming data')
        return self.transform(X, y)
    
class VIFSelector(Selector):
    def __init__(self, threshold=5.0, epsilon=1e-5, verbose=False):
        self.threshold = threshold
        self.epsilon = epsilon
        self.verbose = verbose
        
    def fit(self, X, y):
        '''
        Fits the VIFSelector to the data.
        '''
        if self.verbose:
            print('- Calculating XTX matrix')
        XTX_with_epsilon = np.dot(X.T, X) + self.epsilon * np.eye(X.shape[1])
        if self.verbose:
            print('- Calculating inverse of XTX matrix')
        XTX_inv = np.linalg.inv(XTX_with_epsilon)
        if self.verbose:
            print('- Calculating VIF values')
        self.vif_values = np.diag(XTX_inv) * np.sum(X ** 2, axis=0)
        
    def transform(self, X, y):
        '''
        Transforms the data by removing features with high VIF.
        Returns the transformed data
        '''
        if self.verbose:
            print(f'- Finding columns with high VIF > {self.threshold}')
        high_vif_columns = X.columns[self.vif_values > self.threshold]
        if self.verbose:
            print(f'- Dropping columns with high VIF: {len(high_vif_columns)}')
        return X.drop(columns=high_vif_columns), y
    
class MutualInforegressionSelector(Selector):
    def __init__(self, threshold=0, verbose=False):
        self.threshold = threshold
        self.verbose = verbose
        
    def fit(self, X, y):
        '''
        Fits the selector.
        '''
        self.values = mutual_info_regression(X, y)
        
    def transform(self, X, y):
        '''
        Transforms the data.
        Returns the transformed data
        '''
        return X.loc[:, self.values > self.threshold], y
    
class CooksDistanceInfluenceSelector(Selector):
    def __init__(self, verbose=False):
        super().__init__(verbose)
        
    def fit(self, X, y):
        X_with_const = sm.add_constant(X)
        model = sm.OLS(y.astype(int), np.asarray(X_with_const)).fit()
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        self.influential_points = np.where(cooks_d > 4 / len(y))[0]
        if self.verbose:
            print(f'Identified {len(self.influential_points)} influential points')
    
    def transform(self, X, y):
        X_clean = X.drop(X.index[self.influential_points])
        X_clean = X_clean.loc[:, (X_clean != 0).any(axis=0)]
        y_clean = y.drop(y.index[self.influential_points])
        return X_clean, y_clean
    
class CombineCorrelatedFeaturesSelector(Selector):
    def __init__(self, threshold=0.95, verbose=False):
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y):
        corr_matrix = X.corr().abs()
        correlated_pairs = []
        for i in tqdm(range(len(corr_matrix.columns)), desc='Finding Correlations'):
            for j in range(i):
                if corr_matrix.iloc[i, j] >= self.threshold:
                    correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        # Group the pairs into lists of perfectly correlated features
        correlated_groups = []
        for pair in correlated_pairs:
            added = False
            for group in correlated_groups:
                if pair[0] in group or pair[1] in group:
                    group.update(pair)
                    added = True
                    break
            if not added:
                correlated_groups.append(set(pair))
        # Convert sets to lists for the final output
        self.correlated_features = [list(group) for group in correlated_groups]
    
    def transform(self, X, y):
        reduced_features = {}
        X_df = X.copy()
        for corr_group in tqdm(self.correlated_features, desc='Combining Features'):
            combined_name = '_'.join(corr_group).replace(' ', '') + '_combined'
            reduced_features[combined_name] = PCA(n_components=1).fit_transform(X[corr_group]).reshape(-1)
            X_df.drop(columns=corr_group, inplace=True, errors='ignore')
        X_df = pd.concat([X_df, pd.DataFrame(reduced_features, index=X_df.index)], axis=1)
        return X_df, y