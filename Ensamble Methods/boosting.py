import pandas as pd
import sklearn

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


if __name__ == "__main__":
    
    dt_heart = pd.read_csv('./heart.csv')

    print(dt_heart['target'].describe())

    X = dt_heart.drop(['target'], axis=1)
    y = dt_heart['target']
    
    num_attrs = dt_heart.columns.values
    num_attrs = num_attrs.tolist()
    num_attrs.pop()

    pipeline = ColumnTransformer([("numeric", StandardScaler(), num_attrs)])
    preprocessed_dataset = pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(preprocessed_dataset,y, test_size=0.3)

    boost = GradientBoostingClassifier(n_estimators=50).fit(X_train, y_train)
    boost_pred = boost.predict(X_test)
    print("="*64)
    print(accuracy_score(boost_pred, y_test))

#implementacion_boosting
