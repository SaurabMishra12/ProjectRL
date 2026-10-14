import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import assert_frozen_cycles

class CycleSourcesTests(unittest.TestCase):
    def deliveries(self,order):
        return [{'sender':i,'recipient':(i+1)%3,'text':str(i),'tokens':i+1} for i in order]
    def test_edge_order_does_not_change_sender_payload(self):
        a={'payload_tokens':6,'deliveries':self.deliveries([0,1,2])}
        b={'payload_tokens':6,'deliveries':self.deliveries([0,2,1])}
        assert_frozen_cycles(a,b)
    def test_modified_sender_payload_rejected(self):
        a={'payload_tokens':6,'deliveries':self.deliveries([0,1,2])}
        b={'payload_tokens':6,'deliveries':self.deliveries([0,2,1])}
        b['deliveries'][1]['text']='changed'
        with self.assertRaises(AssertionError):assert_frozen_cycles(a,b)

if __name__=='__main__':unittest.main()
