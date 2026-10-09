import asyncio, importlib.util, unittest
from types import SimpleNamespace
from pathlib import Path
spec=importlib.util.spec_from_file_location('bench',Path(__file__).parents[1]/'benchmark.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

class FakeSearch:
    docs=[{'id':0,'title':'Alpha','text':'Bridge Beta','shard':0}, {'id':1,'title':'Moon','text':'Answer Paris','shard':2}, {'id':2,'title':'Noise','text':'irrelevant','shard':2}]
    def run(self,name,query,topk): return [0] if name=='search_A_F' else [1,2]
class FakeLLM:
    async def call(self,system,payload,usage,role):
        usage.append({'role':role,'seconds':0.1,'input_tokens':10,'output_tokens':5})
        if role=='selector': return {'indices':[j for j,d in enumerate(payload['candidates']) if d['title']!='Noise']}
        if not payload['action_log']: return {'action':'search','calls':[{'tool':'search_A_F','query':'Alpha'},{'tool':'search_M_R','query':'Moon'}]}
        self.final=payload
        return {'action':'finish','answer':'Paris','evidence_ids':[0,1]}
class TestBench(unittest.TestCase):
    def test_flow(self):
        row={'_id':'fixture','question':'Where?','answer':'Paris','supporting_facts':[['Alpha',0],['Moon',0]],'type':'bridge'}
        args=SimpleNamespace(steps=2,rrf_k=60,pool_k=0,max_calls=4,topk=10)
        for mode,expected in [('all',3),('selected',2)]:
            llm=FakeLLM(); result=asyncio.run(b.episode(row,mode,FakeSearch(),llm,args))
            self.assertEqual(result['status'],'ok'); self.assertEqual(result['answer_em'],1)
            self.assertEqual(result['tool_set']['em'],1); self.assertEqual(result['tool_calls'],2)
            self.assertEqual(len(llm.final['evidence']),expected)
            self.assertNotIn('gold_answer',llm.final)
        self.assertEqual(result['final_context_support']['em'],1)
    def test_rrf(self): self.assertEqual(b.fuse([[0,1],[1,2]])[0],1)
    def test_selector(self):
        for vals in [[True],[-1],[3],[1,1]]:
            with self.assertRaises(ValueError): b.validate_indices({'indices':vals},3)
        self.assertEqual(b.validate_indices({'indices':[]},3),[])
    def test_metrics(self):
        self.assertEqual(b.answer_score('The Paris!','Paris'),(1,1.0))
        self.assertEqual(b.answer_score('yes','no'),(0,0.0))
        self.assertEqual(b.set_score(['a','b'],['a'])['precision'],0.5)
if __name__=='__main__': unittest.main()
