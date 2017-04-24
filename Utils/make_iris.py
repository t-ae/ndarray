#!/usr/bin/env python

from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)


def array2str(array2d, spaces):
    ret = []

    for array in array2d:
        ret.append(str(list(array)))

    return (",\n" + " "*spaces).join(ret)

data = f"""
import NDArray

struct Iris {{
    static let x_train = NDArray([
        {array2str(x_train, 8)}
    ])
    static let y_train = NDArray({list(y_train)})
    
    static let x_test = NDArray([
        {array2str(x_test, 8)}
    ])
    static let y_test = NDArray({list(y_test)})
}}
"""[1:-1]

print(data)
