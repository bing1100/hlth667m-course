import math
import torch
import pytest

def attention(query,key,value,allowed_mask=None):
 if query.shape[-1]!=key.shape[-1]: raise ValueError('Query/key dimensions must match.')
 if key.shape[-2]!=value.shape[-2]: raise ValueError('Key/value sequence lengths must match.')
 scores=query@key.transpose(-2,-1)/math.sqrt(query.shape[-1])
 if allowed_mask is not None:
  if torch.any(~allowed_mask.any(dim=-1)): raise ValueError('Every query row needs an allowed key.')
  scores=scores.masked_fill(~allowed_mask,-torch.inf)
 weights=torch.softmax(scores,-1); return weights@value,weights,scores

def fixture():
 q=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64)
 return q,q.clone(),torch.tensor([[10.,0.],[0.,20.]],dtype=torch.float64)

def test_hand_fixture_and_causal_mask():
 q,k,v=fixture(); out,w,_=attention(q,k,v)
 torch.testing.assert_close(w,torch.tensor([[.66976155,.33023845],[.33023845,.66976155]],dtype=torch.float64),rtol=1e-6,atol=1e-7)
 torch.testing.assert_close(out,torch.tensor([[6.69761549,6.60476902],[3.30238451,13.39523098]],dtype=torch.float64),rtol=1e-6,atol=1e-7)
 _,masked,_=attention(q,k,v,torch.tril(torch.ones(2,2,dtype=torch.bool)))
 assert torch.allclose(masked.sum(-1),torch.ones(2,dtype=torch.float64))
 assert masked[0,1] == 0

def test_rejects_invalid_shapes_and_fully_masked_row():
 q,k,v=fixture()
 with pytest.raises(ValueError): attention(q,torch.ones(2,3,dtype=torch.float64),v)
 with pytest.raises(ValueError): attention(q,k,v,torch.tensor([[False,False],[True,True]]))
