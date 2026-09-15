from sklearn.model_selection import train_test_split

from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.neighbors import KNeighborsRegressor
# import other regressors
# import support vectors, bayes
# import ensemble methods
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.naive_bayes import GaussianNB


from tqdm.notebook import tqdm


class PreprocessingValidator:
    # Prints the preprocessing validation results
    def __init__(self, steps, verbose=False):
        self.steps = steps
        self.step_name = None
        self.verbose = verbose
        
    def validate(self, X, y):
        '''
        Validates the preprocessing steps.
        '''
        self.X = X
        self.y = y
        results = {'baseline': self.test_step(X, y)}
        shapes = {'baselline': X.shape}
        for step in self.steps:
            # Drop columns where all values are 0
            self.X = self.X.loc[:, (self.X != 0).any(axis=0)]
            self.step_name = step.__class__.__name__
            self.X, self.y = step.fit_transform(self.X, self.y)
            results.update({step.__class__.__name__: self.test_step(self.X, self.y)})
            shapes.update({step.__class__.__name__: self.X.shape})
        self.results = results
        self.shapes = shapes
        print('Validation complete.')
            
    def test_step(self, step_X, step_y):
        '''
        Tests a step.
        '''
        models = [
            Lasso(), 
            ElasticNet(), 
            ]
        X_train, X_test, y_train, y_test = train_test_split(step_X, step_y, test_size=.1)
        performance = {}
        pbar = tqdm(total=len(models))
        for model in models:
            pbar.set_description(f'Validating {self.step_name} with {model.__class__.__name__}')
            model.fit(X_train, y_train)
            performance[model.__class__.__name__] = model.score(X_test, y_test)
            pbar.update(1)
        return performance
    
    def get_results(self):
        '''
        Returns the results.
        '''
        
        return self.results
    
    def get_shapes(self):
        '''
        Returns the shapes.
        '''
        return self.shapes