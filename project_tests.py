import unittest
from project import gradient_test1
from project import gradient_test2
from project import gradient_test3
from project import differentiable_function_test
from project import gradient_search_test
from project import silvestr_criterion
from project import derivative_2_test



class ProjectTests(unittest.TestCase):
    def test1(self):
        self.assertEqual(gradient_test1([7],0.00000001), [17.0])

    def test2(self):
        self.assertEqual(gradient_test2([2,9],0.00000001), [4.0, 18.0])

    def test3(self):
        self.assertEqual(gradient_test3([2,3,4],0.00000001), [-2899999971.0, -2899999971.0, -2899999971.0])

    def test4(self):
        self.assertEqual(differentiable_function_test([2,4]), 20)

    def test5(self):
        self.assertEqual(gradient_search_test(x0 = [0], dx = 0.0001 ,center = [0], radius = 8, extr_type='min', epsilon=10**(-4)), -1.5)

    def test6(self):
        self.assertEqual(silvestr_criterion([0]), "Not an extremum")

    def test7(self):
        self.assertEqual(derivative_2_test([-1,5,6], dx=0.0001), [[20000.0, 20000.0, 20000.0], [-99990.0, -99990.0, -99990.0], [-119990.0, -119990.0, -119990.0]])

if __name__ == "__main__":
    unittest.main()