"""Small structural fixtures are unit tests, never experimental datasets."""
import sys
import unittest
from pathlib import Path
from itertools import combinations
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.gold_strata import classify_gold, heterogeneity

class GoldStrataTests(unittest.TestCase):
    def row(self,a='A',b='B'):
        return {'id':'fixture','type':'bridge','supporting_facts':[[a,0],[b,0],[a,1]]}
    def test_gold_locations(self):
        self.assertEqual(classify_gold(self.row(),[[{'title':'A'}],[{'title':'B'}],[]])['stratum'],'Disjoint-Gold')
        self.assertEqual(classify_gold(self.row(),[[{'title':'A'},{'title':'B'}],[],[]])['stratum'],'Co-located-Gold')
        self.assertEqual(classify_gold(self.row(),[[{'title':'A'}],[],[]])['stratum'],'Retrieval-Missing-Gold')
        self.assertEqual(classify_gold(self.row('A','A'),[[{'title':'A'}],[],[]])['stratum'],'Undefined-Gold')
    def test_exclusive_shards_required(self):
        with self.assertRaises(ValueError):
            classify_gold(self.row(),[[{'title':'A'}],[{'title':'A'},{'title':'B'}],[]])
    def test_exact_combinatorics(self):
        for sizes,expected,total in [([3,3,4],12,45),([2,2,2],3,15)]:
            owners=[i for i,n in enumerate(sizes) for _ in range(n)]
            colocated=sum(owners[a]==owners[b] for a,b in combinations(range(len(owners)),2))
            self.assertEqual(colocated,expected)
            self.assertEqual(len(list(combinations(range(len(owners)),2))),total)
    def test_heterogeneity_empty_and_constant(self):
        self.assertEqual(heterogeneity({}, {'a':1})['status'],'UNESTIMABLE_EMPTY_STRATUM')
        result=heterogeneity({'a':3,'b':3},{'c':1,'d':1},draws=20)
        self.assertEqual(result['effect'],2)
        self.assertEqual(result['ci95'],[2,2])

if __name__=='__main__':unittest.main()
