from unittest import TestCase
import os
from text.dataset import Example
from text.vocab import Vocab
from dataset import ExampleAdaptor

__author__ = 'victor'

mydir = os.path.dirname(__file__)

ex = {"confidence": "1.0000", "entityCharOffsetEnd": "118", "sentence": "With The Last Victim of the White Slave Trade ( Den Hvide Slavehandels Sidste Offer ) ( 1911 ) Directed by August Blom One of the many &amp;quot; white slave trade &amp;quot; films popular in Denmark &#44; ( which dealt with young women being sold to brothels ) the film is wonderfully exciting &#44; especially the climactic fight on a rooftop .", "jsDistance": "0.8898", "entityCharOffsetBegin": "106", "slotValue": " Denmark", "entity": " August Blom", "slotValueCharOffsetEnd": "199", "relation": "no_relation", "slotValueCharOffsetBegin": "191", "key": "1e14aba0201ed046121760e8a0886be262f19dd6a7189ce112ff44b0470c67a5:21-23:35-36"}

class TestExampleAdaptor(TestCase):
    def setUp(self):
        self.word_vocab, self.rel_vocab = Vocab(), Vocab()
        self.word_vocab.add('UNKNOWN')
        self.ex = Example(ex)
        for w in self.ex.sentence.lower().split():
            self.word_vocab.add(w)
        self.adaptor = ExampleAdaptor(self.word_vocab, self.rel_vocab)

    def test_convert(self):
        expect = [
            (-1, "With The Last Victim of the White Slave Trade ( Den Hvide Slavehandels Sidste Offer ) ( 1911 ) Directed by August Blom One of the many &amp;quot; white slave trade &amp;quot; films popular in Denmark &#44; ( which dealt with young women being sold to brothels ) the film is wonderfully exciting &#44; especially the climactic fight on a rooftop ."),
            (0, "August Blom One of the many &amp;quot; white slave trade &amp;quot; films popular in Denmark"),
            (2, "Directed by August Blom One of the many &amp;quot; white slave trade &amp;quot; films popular in Denmark &#44; (")
        ]

        for context, sent in expect:
            got = self.adaptor.convert(self.ex, context)
            got_rel = self.rel_vocab.index2word[got.relation]
            got_sent = ' '.join([self.word_vocab.index2word[w] for w in got.sentence])
            rel = ex[u'relation']
            msg = "for context " + str(context) + "\n" + ' '.join([str(s) for s in ['Expected:', sent.lower().split(), "\n", 'Actual:', got_sent.split()]])
            self.assertEqual(sent.lower(), got_sent, msg=msg)
            self.assertEqual(rel, got_rel)
