"""Drive-resumable CUDA experiments for the SMILES MIA project.

All generated experiment files live below the Drive project folder. This file
is intended to run from the packaged project root in Google Colab.
"""
from __future__ import annotations
import json, os, random, shutil, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from rdkit import Chem
from sklearn.metrics import roc_curve

from src.data import load_dataset
from src.model import GPT
from src.tokenizer import Tokenizer
from src.attack import attack_table
from src.train import sequence_nll

ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))
ARM = 'CUDA'
N_SHADOWS = 32
EPOCHS = 60
POOL_N = 20000
VAL_N = 5000
MEMBER_SIZES = (1000, 2500, 5000, 10000)
DELTA = 1e-5

def set_root(path: str | Path):
    global ROOT
    ROOT = Path(path).resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["PROJECT_ROOT"] = str(ROOT)
    print(f"Project root set to: {ROOT}")
    return ROOT

def _atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)

def _manifest():
    p = ROOT / 'results' / 'cuda_arm' / 'manifest.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists(): return p, json.loads(p.read_text())
    data = {'arm': ARM, 'device': str(torch.cuda.get_device_name(0)), 'stages': {}}
    _atomic_json(p, data)
    return p, data

def _stage_status(stage, state, **fields):
    p, data = _manifest()
    entry = data['stages'].setdefault(stage, {})
    entry.update({'state': state, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields})
    _atomic_json(p, data)

def _prepare_data():
    d = ROOT / 'data'
    required = [d/'pool.csv', d/'val.csv', d/'pool_metadata.csv']
    if all(p.exists() for p in required):
        pool = pd.read_csv(required[0]).smiles.tolist()
        val = pd.read_csv(required[1]).smiles.tolist()
        meta = pd.read_csv(required[2])
        if len(pool) == 35000 and len(val) == VAL_N and len(meta) == 35000:
            return pool, val, meta
    d.mkdir(parents=True, exist_ok=True)
    raw = d/'moses.csv'
    if not raw.exists():
        import requests
        url = 'https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv'
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = raw.with_suffix('.csv.partial')
            with tmp.open('wb') as f:
                for chunk in r.iter_content(1024*1024):
                    if chunk: f.write(chunk)
            os.replace(tmp, raw)
    # Extra disjoint public molecules are reserved for the pretrain/fine-tune scenario.
    pool, val = load_dataset(path=str(raw), pool_size=35000, val_size=VAL_N, seed=0)
    for name in ('pool.csv', 'val.csv', 'pool_metadata.csv'):
        tmp = d / (name + '.tmp')
        shutil.copy2(d/name, tmp)
        os.replace(tmp, d/name)
    meta = pd.read_csv(d/'pool_metadata.csv')
    return pool, val, meta

def _tokenizer(pool, val, meta):
    variants = sum((s.split('|') for s in meta.variants), [])
    return Tokenizer(pool + val + variants)

def _encode_batch(strings, tok):
    return [torch.tensor(tok.encode(s)[:128], dtype=torch.long) for s in strings]

def _build_model(vocab_size, dp_compatible=False):
    model = GPT(vocab_size, n_layer=2, d_model=128)
    if dp_compatible:
        # Opacus' DPMultiheadAttention currently rejects causal masks, so the
        # DP arm uses equivalent causal attention built from supported modules.
        from opacus.validators import ModuleValidator
        from src.dp_model import DPCausalGPT
        model = DPCausalGPT(vocab_size, n_layer=2, d_model=128)
        ModuleValidator.validate(model, strict=True)
    return model.cuda()

def _train_shadow(stage, k, train_strings, val, tok, epochs=EPOCHS,
                  augment=False, init_state=None, dp_epsilon=None, seed=None):
    seed = int(k if seed is None else seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    d = ROOT/'runs'/'cuda_arm'/stage/f'model_{k}'
    d.mkdir(parents=True, exist_ok=True)
    cp = d/'latest.pt'
    final_cp = d/f'epoch{epochs}.pt'
    
    # If final epoch checkpoint and loss log already exist, skip training
    if final_cp.exists() and (d/'losses.csv').exists():
        try:
            losses_df = pd.read_csv(d/'losses.csv')
            if len(losses_df) >= epochs:
                print(f"[{stage}] Model {k} already completed ({len(losses_df)}/{epochs} epochs). Skipping.")
                return None
        except Exception:
            pass

    model = _build_model(len(tok.itos), dp_compatible=dp_epsilon is not None)
    if init_state is not None: model.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    start_epoch, logs = 0, []
    if cp.exists():
        state = torch.load(cp, map_location='cuda', weights_only=False)
        start_epoch, logs = state['epoch'], state.get('logs', [])
        print(f"[{stage}] Model {k} found checkpoint at epoch {start_epoch}/{epochs}. Resuming from epoch {start_epoch + 1}...")
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        # Older/runtime-specific checkpoints can deserialize RNG states with
        # a non-byte dtype. PyTorch requires CPU uint8 tensors for restoration.
        # Use the legacy ByteTensor class explicitly: some Colab/PyTorch builds
        # reject a generic uint8 Tensor at the RNG restore boundary.
        torch_rng = torch.as_tensor(state['torch_rng']).detach().cpu().to(torch.uint8).type(torch.ByteTensor)
        cuda_rng = [torch.as_tensor(s).detach().cpu().to(torch.uint8).type(torch.ByteTensor)
                    for s in state['cuda_rng']]
        torch.set_rng_state(torch_rng); torch.cuda.set_rng_state_all(cuda_rng)
    else:
        print(f"[{stage}] Training model {k} from scratch (epoch 1/{epochs})...")
    private_model, privacy_engine, train_loader = model, None, None
    if dp_epsilon is not None:
        from opacus import PrivacyEngine
        from opacus.accountants.utils import get_noise_multiplier
        base = pad_sequence(_encode_batch(train_strings, tok), batch_first=True, padding_value=tok.pad_id)
        train_loader = DataLoader(TensorDataset(base), batch_size=32, shuffle=True, drop_last=False)
        sigma = get_noise_multiplier(target_epsilon=float(dp_epsilon), target_delta=DELTA,
                                     sample_rate=min(1.0, 32/len(train_strings)), epochs=epochs)
        privacy_engine = PrivacyEngine()
        private_model, opt, train_loader = privacy_engine.make_private(
            module=model, optimizer=opt, data_loader=train_loader,
            noise_multiplier=sigma, max_grad_norm=1.0,
        )
        model = private_model
        # Private optimizer/accountant are recreated on restart; weights and epoch resume.
        # Manifest records that epsilon accounting restarts only if accountant state is unavailable.
        if cp.exists() and state.get('privacy_accountant'):
            privacy_engine.accountant.load_state_dict(state['privacy_accountant'])
    vx = pad_sequence(_encode_batch(val, tok), batch_first=True, padding_value=tok.pad_id).cuda()
    for ep in range(start_epoch + 1, epochs + 1):
        model.train(); total=0.0; batches=0
        if dp_epsilon is None:
            current = train_strings
            if augment:
                current = []
                for s in train_strings:
                    mol = Chem.MolFromSmiles(s)
                    current.append(Chem.MolToSmiles(mol, canonical=False, doRandom=True) if mol else s)
            order = torch.randperm(len(current))
            batches_iter = (order[i:i+32].tolist() for i in range(0, len(order), 32))
            for ix in batches_iter:
                bx = pad_sequence(_encode_batch([current[j] for j in ix], tok), batch_first=True, padding_value=tok.pad_id).cuda()
                opt.zero_grad(set_to_none=True)
                logits = model(bx[:,:-1])
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), bx[:,1:].reshape(-1), ignore_index=tok.pad_id)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                total += float(loss.detach()); batches += 1
        else:
            for (bx,) in train_loader:
                bx=bx.cuda(); opt.zero_grad(set_to_none=True)
                logits=model(bx[:,:-1]); loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.size(-1)),bx[:,1:].reshape(-1),ignore_index=tok.pad_id)
                loss.backward(); opt.step(); total += float(loss.detach()); batches += 1
        model.eval()
        with torch.no_grad():
            val_loss = torch.nn.functional.cross_entropy(model(vx[:,:-1]).reshape(-1,len(tok.itos)),vx[:,1:].reshape(-1),ignore_index=tok.pad_id).item()
        rec={'epoch':ep,'train_loss':total/max(1,batches),'val_loss':val_loss}
        if privacy_engine is not None: rec['epsilon_spent']=privacy_engine.get_epsilon(DELTA)
        logs.append(rec)
        state_model = model._module if hasattr(model, '_module') else model
        payload={'model':state_model.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'logs':logs,
                 'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                 'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                 'arm':ARM,'stage':stage,'seed':seed,'dp_epsilon':dp_epsilon}
        if privacy_engine is not None: payload['privacy_accountant']=privacy_engine.accountant.state_dict()
        tmp=cp.with_suffix('.pt.tmp'); torch.save(payload,tmp); os.replace(tmp,cp)
        if ep in (5,20,60):
            tmp=d/f'epoch{ep}.pt.tmp'; torch.save({'model':state_model.state_dict(),'epoch':ep,'arm':ARM},tmp); os.replace(tmp,d/f'epoch{ep}.pt')
        pd.DataFrame(logs).to_csv(d/'losses.csv.tmp',index=False); os.replace(d/'losses.csv.tmp',d/'losses.csv')
    return model

def _membership(pool, k, n_pool=None):
    n = len(pool) if n_pool is None else n_pool
    rng=np.random.default_rng(k)
    idx=rng.choice(n,n//2,replace=False)
    mask=np.zeros(n,dtype=bool); mask[idx]=True
    return mask

def _write_split_control(stage, pool, val, masks):
    """Record C3 split/mask checks after explicit validation."""
    masks=np.asarray(masks,dtype=bool); n=len(pool)
    records=[{'check':'pool_val_disjoint','model':'','pool_size':n,
              'member_count':'','overlap_count':len(set(pool)&set(val)),
              'pass':not (set(pool)&set(val))}]
    shape_ok=masks.ndim==2 and masks.shape[1]==n
    for k,mask in enumerate(masks):
        members=set(np.flatnonzero(mask)); out=set(range(n))-members
        records.append({'check':'balanced_member_mask','model':k,'pool_size':n,
                        'member_count':len(members),'overlap_count':len(members & out),
                        'pass':shape_ok and len(members)==n//2 and not (members & out)
                              and len(members | out)==n})
    out=ROOT/'results'/'cuda_arm'/stage/'split_control.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(out,index=False)
    if not all(bool(row['pass']) for row in records):
        raise ValueError(f'C3 split/mask validation failed for {stage}; see {out}')

def _score_shadows(stage, pool, val, meta, tok, masks, epochs=(5,20,60), variants_at_60=False):
    outdir=ROOT/'results'/'cuda_arm'/stage; outdir.mkdir(parents=True,exist_ok=True)
    for k, mask in enumerate(masks):
        md=ROOT/'runs'/'cuda_arm'/stage/f'model_{k}'
        for ep in epochs:
            out=outdir/f'model_{k}_epoch{ep}.parquet'
            if out.exists(): continue
            cp=md/f'epoch{ep}.pt'
            state=torch.load(cp,map_location='cuda',weights_only=False)
            model=GPT(len(tok.itos),n_layer=2,d_model=128).cuda(); model.load_state_dict(state['model']); model.eval()
            values=sequence_nll(model,tok,pool,'cuda')
            df=pd.DataFrame({'id':np.arange(len(pool)),'nll_sum':[x[0] for x in values],'nll_mean':[x[1] for x in values]})
            if variants_at_60 and ep==60:
                vv=[]
                variant_rows=[row.split('|') for row in meta.variants]
                for start in range(0,len(variant_rows),128):
                    group=variant_rows[start:start+128]
                    flat=[s for ss in group for s in ss]
                    scores=sequence_nll(model,tok,flat,'cuda')
                    offset=0
                    for ss in group:
                        vv.append(float(np.mean([x[0] for x in scores[offset:offset+len(ss)]])))
                        offset += len(ss)
                df['nll_variants_mean']=vv
            tmp=out.with_suffix('.parquet.tmp'); df.to_parquet(tmp,index=False); os.replace(tmp,out)
            del model, state, values
            torch.cuda.empty_cache()

def _summarize(stage, masks, epochs, stats=('nll_sum','nll_mean')):
    base=ROOT/'results'/'cuda_arm'/stage; rows=[]
    for ep in epochs:
        ds=[pd.read_parquet(base/f'model_{k}_epoch{ep}.parquet') for k in range(len(masks))]
        for stat in stats:
            arr=np.stack([d[stat].to_numpy() for d in ds])
            tbl=attack_table(arr,masks,stat=stat); tbl['epoch']=ep; tbl['arm']=ARM; rows.append(tbl)
        if 'nll_variants_mean' in ds[0].columns:
            arr=np.stack([d.nll_variants_mean.to_numpy() for d in ds])
            tbl=attack_table(arr,masks,stat='nll_variants_mean'); tbl['epoch']=ep; tbl['arm']=ARM; rows.append(tbl)
    allrows=pd.concat(rows,ignore_index=True); allrows.to_csv(base/'attacks_by_target.csv',index=False)
    summary=allrows.groupby(['arm','epoch','stat','method']).agg(
        auc_mean=('auc','mean'),auc_std=('auc','std'),tpr1_mean=('tpr_fpr_1pct','mean'),tpr1_std=('tpr_fpr_1pct','std'),
        tpr01_mean=('tpr_fpr_0_1pct','mean'),tpr01_std=('tpr_fpr_0_1pct','std'),balanced_accuracy_mean=('balanced_accuracy','mean'),
        balanced_accuracy_std=('balanced_accuracy','std')).reset_index()
    summary.to_csv(base/'attack_summary.csv',index=False)

def run_cuda_baseline(pool, val, meta, tok):
    stage='baseline'; pool=pool[:POOL_N]; meta=meta.iloc[:POOL_N]; masks=[]
    for k in range(N_SHADOWS):
        mask=_membership(pool,k); masks.append(mask)
        md=ROOT/'runs'/'cuda_arm'/stage/f'model_{k}'; md.mkdir(parents=True,exist_ok=True)
        np.save(md/'membership_mask.npy',mask)
        _train_shadow(stage,k,[pool[i] for i in np.flatnonzero(mask)],val,tok,epochs=EPOCHS,seed=k)
    _write_split_control(stage,pool,val,masks)
    _score_shadows(stage,pool[:POOL_N],val,meta.iloc[:POOL_N],tok,masks,epochs=(5,20,60),variants_at_60=True)
    _summarize(stage,masks,(5,20,60))
    _stage_status(stage,'complete',shadows=N_SHADOWS,pool=POOL_N,epochs=EPOCHS)

def run_e3(pool, val, tok):
    stage='E3'; root=ROOT/'results'/'cuda_arm'/stage; root.mkdir(parents=True,exist_ok=True); summaries=[]
    for size in MEMBER_SIZES:
        universe=pool[:2*size]; masks=[]
        for k in range(8):
            mask=_membership(universe,k); masks.append(mask)
            d=ROOT/'runs'/'cuda_arm'/stage/f'size_{size}'/f'model_{k}'; d.mkdir(parents=True,exist_ok=True)
            np.save(d/'membership_mask.npy',mask)
            _train_shadow(f'E3/size_{size}',k,[universe[i] for i in np.flatnonzero(mask)],val,tok,epochs=60,seed=10000+size+k)
            # Keep staged directory name aligned with resume implementation.
            src=ROOT/'runs'/'cuda_arm'/f'E3/size_{size}'/f'model_{k}'
            if src.exists() and src != d: shutil.copytree(src,d,dirs_exist_ok=True)
        _write_split_control(f'E3/size_{size}',universe,val,masks)
        # Score and attack for this fixed pool size.
        scores=[]
        for k in range(8):
            d=ROOT/'runs'/'cuda_arm'/f'E3/size_{size}'/f'model_{k}'
            cp=d/'epoch60.pt'; state=torch.load(cp,map_location='cuda',weights_only=False)
            model=GPT(len(tok.itos),n_layer=2,d_model=128).cuda(); model.load_state_dict(state['model'])
            vals=sequence_nll(model,tok,universe,'cuda'); scores.append([x[0] for x in vals]); del model; torch.cuda.empty_cache()
        tbl=attack_table(np.asarray(scores),np.asarray(masks),stat='nll_sum'); tbl['member_size']=size; tbl['arm']=ARM; summaries.append(tbl)
    allrows=pd.concat(summaries,ignore_index=True); allrows.to_csv(root/'attacks_by_target.csv',index=False)
    allrows.groupby(['arm','member_size','method']).agg(auc_mean=('auc','mean'),auc_std=('auc','std'),tpr1_mean=('tpr_fpr_1pct','mean'),tpr1_std=('tpr_fpr_1pct','std')).reset_index().to_csv(root/'summary.csv',index=False)
    _stage_status(stage,'complete',member_sizes=MEMBER_SIZES,shadows_per_size=8)

def run_e5a(pool,val,meta,tok):
    stage='E5a'; pool=pool[:POOL_N]; meta=meta.iloc[:POOL_N]; masks=[]
    for k in range(N_SHADOWS):
        mask=_membership(pool,k); masks.append(mask)
        md=ROOT/'runs'/'cuda_arm'/stage/f'model_{k}'; md.mkdir(parents=True,exist_ok=True); np.save(md/'membership_mask.npy',mask)
        _train_shadow(stage,k,[pool[i] for i in np.flatnonzero(mask)],val,tok,epochs=EPOCHS,augment=True,seed=20000+k)
    _write_split_control(stage,pool,val,masks)
    _score_shadows(stage,pool[:POOL_N],val,meta.iloc[:POOL_N],tok,masks,epochs=(5,20,60),variants_at_60=True)
    _summarize(stage,masks,(5,20,60))
    _stage_status(stage,'complete',shadows=N_SHADOWS,epochs=EPOCHS,training='randomized-SMILES augmentation')

def run_pretrain_ft(pool,val,tok):
    stage='pretrain_ft'; pretrain_data=pool[POOL_N:POOL_N+10000]; private_pool=pool[:POOL_N]
    if len(pretrain_data)<10000: raise ValueError('Need 10k reserved public pretraining molecules')
    pre=ROOT/'runs'/'cuda_arm'/stage/'pretrain'; pre.mkdir(parents=True,exist_ok=True)
    _train_shadow(stage+'/pretrain',0,pretrain_data,val,tok,epochs=60,seed=30000)
    state=torch.load(pre/'model_0'/'epoch60.pt',map_location='cuda',weights_only=False)['model']
    masks=[]
    for k in range(8):
        mask=_membership(private_pool,k); masks.append(mask)
        d=ROOT/'runs'/'cuda_arm'/stage/f'model_{k}'; d.mkdir(parents=True,exist_ok=True); np.save(d/'membership_mask.npy',mask)
        _train_shadow(stage,k,[private_pool[i] for i in np.flatnonzero(mask)],val,tok,epochs=20,init_state=state,seed=31000+k)
    _write_split_control(stage,private_pool,val,masks)
    scores=[]
    for k in range(8):
        state=torch.load(ROOT/'runs'/'cuda_arm'/stage/f'model_{k}'/'epoch20.pt',map_location='cuda',weights_only=False)
        model=GPT(len(tok.itos),n_layer=2,d_model=128).cuda(); model.load_state_dict(state['model'])
        vals=sequence_nll(model,tok,private_pool,'cuda'); scores.append([x[0] for x in vals]); del model; torch.cuda.empty_cache()
    tbl=attack_table(np.asarray(scores),np.asarray(masks),stat='nll_sum'); tbl['arm']=ARM; tbl['scenario']='public pretrain then private fine-tune'; tbl.to_csv(ROOT/'results'/'cuda_arm'/stage/'attacks_by_target.csv',index=False)
    _stage_status(stage,'complete',pretraining_molecules=len(pretrain_data),private_pool=len(private_pool),shadows=8)

def run_dp_sgd(pool,val,tok):
    stage='DP_SGD'; pool=pool[:POOL_N]; root=ROOT/'results'/'cuda_arm'/stage; root.mkdir(parents=True,exist_ok=True); rows=[]
    privacy_rows=[]
    for eps in (8,3):
        masks=[]; scores=[]
        for k in range(8):
            mask=_membership(pool,k); masks.append(mask)
            d=ROOT/'runs'/'cuda_arm'/stage/f'eps_{eps}'/f'model_{k}'; d.mkdir(parents=True,exist_ok=True); np.save(d/'membership_mask.npy',mask)
            _train_shadow(f'DP_SGD/eps_{eps}',k,[pool[i] for i in np.flatnonzero(mask)],val,tok,epochs=60,dp_epsilon=eps,seed=40000+eps*100+k)
            losses=pd.read_csv(ROOT/'runs'/'cuda_arm'/stage/f'eps_{eps}'/f'model_{k}'/'losses.csv')
            privacy_rows.append({'arm':ARM,'epsilon_target':eps,'delta':DELTA,'model':k,
                                 'epsilon_spent_last_epoch':float(losses.epsilon_spent.iloc[-1])})
            state=torch.load(d/'epoch60.pt',map_location='cuda',weights_only=False)
            model=_build_model(len(tok.itos),dp_compatible=True); model.load_state_dict(state['model'])
            vals=sequence_nll(model,tok,pool,'cuda'); scores.append([x[0] for x in vals]); del model; torch.cuda.empty_cache()
        _write_split_control(f'{stage}/eps_{eps}',pool,val,masks)
        tbl=attack_table(np.asarray(scores),np.asarray(masks),stat='nll_sum'); tbl['epsilon_target']=eps; tbl['arm']=ARM; rows.append(tbl)
    allrows=pd.concat(rows,ignore_index=True); allrows.to_csv(root/'attacks_by_target.csv',index=False)
    allrows.groupby(['arm','epsilon_target','method']).agg(auc_mean=('auc','mean'),auc_std=('auc','std'),tpr1_mean=('tpr_fpr_1pct','mean'),tpr1_std=('tpr_fpr_1pct','std')).reset_index().to_csv(root/'summary.csv',index=False)
    pd.DataFrame(privacy_rows).to_csv(root/'privacy_accounting_by_model.csv',index=False)
    _stage_status(stage,'complete',epsilon_targets=[8,3],shadows_per_epsilon=8)

def _generate(model,tok,n,batch_size=128,temperature=1.0,max_len=128,checkpoint_dir=None):
    out=[]; cp=Path(checkpoint_dir) if checkpoint_dir else None
    if cp: cp.parent.mkdir(parents=True, exist_ok=True)
    if cp and cp.exists():
        saved=torch.load(cp,map_location='cpu',weights_only=False); out=saved['smiles']; start=len(out)
    else: start=0
    dev = next(model.parameters()).device
    while start<n:
        b=min(batch_size,n-start); seq=torch.full((b,1),tok.bos_id,dtype=torch.long,device=dev); done=torch.zeros(b,dtype=torch.bool,device=dev)
        for _ in range(max_len-1):
            logits=model(seq[:,-model.context:])[:,-1,:]/temperature
            nxt=torch.multinomial(torch.softmax(logits,dim=-1),1).squeeze(1)
            nxt=torch.where(done,torch.full_like(nxt,tok.pad_id),nxt)
            seq=torch.cat([seq,nxt[:,None]],dim=1); done |= nxt.eq(tok.eos_id)
            if bool(done.all()): break
        out.extend([tok.decode(row.tolist()) for row in seq.cpu()])
        start=len(out)
        if cp:
            tmp=cp.with_suffix('.tmp'); torch.save({'smiles':out,'cursor':start},tmp); os.replace(tmp,cp)
    return out[:n]

def run_e6(pool,tok):
    stage='E6'; root=ROOT/'results'/'cuda_arm'/stage; root.mkdir(parents=True,exist_ok=True); summaries=[]
    dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    for k in (0,15,31):
        d=ROOT/'runs'/'cuda_arm'/'baseline'/f'model_{k}'
        state=torch.load(d/'epoch60.pt',map_location=dev,weights_only=False)
        model=GPT(len(tok.itos),n_layer=2,d_model=128).to(dev); model.load_state_dict(state['model']); model.eval()
        mask=np.load(d/'membership_mask.npy'); member={pool[i] for i in np.flatnonzero(mask)}; out={pool[i] for i in np.flatnonzero(~mask)}
        generations=_generate(model,tok,5000,temperature=1.0,checkpoint_dir=ROOT/'runs'/'cuda_arm'/stage/f'model_{k}_generated.pt')
        canonical=[]; valid=[]
        for s in generations:
            m=Chem.MolFromSmiles(s)
            valid.append(m is not None)
            canonical.append(Chem.MolToSmiles(m,canonical=True) if m is not None else '')
        valid_smiles=[s for s,v in zip(canonical,valid) if v]
        member_hits=sum(s in member for s in valid_smiles); out_hits=sum(s in out for s in valid_smiles)
        summaries.append({'arm':ARM,'model':k,'samples':len(generations),'validity':float(np.mean(valid)),
            'uniqueness_valid':len(set(valid_smiles))/max(1,len(valid_smiles)),
            'novelty_vs_members':float(np.mean([s not in member for s in valid_smiles])) if valid_smiles else np.nan,
            'exact_member_copy_fraction_all_samples':member_hits/len(generations),
            'exact_out_copy_fraction_all_samples':out_hits/len(generations)})
        pd.DataFrame({'smiles':generations,'canonical':canonical,'valid':valid}).to_parquet(root/f'model_{k}_samples.parquet',index=False)
        del model
        if dev == 'cuda': torch.cuda.empty_cache()
    pd.DataFrame(summaries).to_csv(root/'summary.csv',index=False)
    _stage_status(stage,'complete',models=[0,15,31],samples_per_model=5000)

def preflight():
    """Check GPU, data preparation, and Opacus compatibility without training."""
    from opacus import PrivacyEngine, __version__ as opacus_version
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA runtime is required; select a Colab GPU first.')
    device = torch.cuda.get_device_name(0)
    rows = [
        {'check':'cuda','status':'pass','detail':device},
        {'check':'torch','status':'pass','detail':torch.__version__},
        {'check':'opacus','status':'pass','detail':opacus_version},
    ]
    try:
        # Tiny synthetic backward pass exercises Opacus grad-sample hooks, but
        # deliberately does not call optimizer.step() or update model weights.
        test_model = _build_model(64, dp_compatible=True)
        test_optimizer = torch.optim.AdamW(test_model.parameters(), lr=1e-3)
        sample = torch.randint(0, 64, (2, 8), dtype=torch.long)
        test_loader = DataLoader(TensorDataset(sample), batch_size=2, shuffle=False)
        engine = PrivacyEngine()
        private_model, private_optimizer, private_loader = engine.make_private(
            module=test_model, optimizer=test_optimizer, data_loader=test_loader,
            noise_multiplier=1.0, max_grad_norm=1.0, poisson_sampling=False,
        )
        (batch,) = next(iter(private_loader))
        logits = private_model(batch[:, :-1].cuda())
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), batch[:, 1:].cuda().reshape(-1)
        )
        loss.backward()
        missing = [name for name, param in private_model.named_parameters()
                   if param.requires_grad and not hasattr(param, 'grad_sample')]
        if missing:
            raise RuntimeError(f'Opacus did not create per-sample gradients for: {missing[:8]}')
        private_optimizer.zero_grad(set_to_none=True)
        rows.append({'check':'opacus_dp_model_backward','status':'pass',
                     'detail':'synthetic backward only; optimizer.step was not called'})
        del private_model, private_optimizer, private_loader, test_model, test_optimizer
        torch.cuda.empty_cache()

        pool, val, meta = _prepare_data()
        _tokenizer(pool, val, meta)
        if len(pool) != 35000 or len(val) != VAL_N or len(meta) != 35000:
            raise ValueError(f'Unexpected prepared data sizes: pool={len(pool)}, val={len(val)}, metadata={len(meta)}')
        if set(pool) & set(val):
            raise ValueError('Prepared POOL and VAL overlap.')
        rows.extend([
            {'check':'dataset_pool','status':'pass','detail':len(pool)},
            {'check':'dataset_val','status':'pass','detail':len(val)},
            {'check':'pool_val_disjoint','status':'pass','detail':'zero overlap'},
        ])
        out = ROOT/'results'/'cuda_arm'/'preflight.csv'
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        _stage_status('preflight','complete',device=device,opacus=opacus_version,
                      pool=len(pool),val=len(val),dp_smoke='passed_without_optimizer_step')
        print(f'Preflight passed; details saved to {out}')
        return out
    except Exception as exc:
        _stage_status('preflight','error',device=device,error=repr(exc))
        raise

def run_stages(stages):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA runtime is required for the CUDA arm; select a Colab GPU runtime.')
    device=torch.cuda.get_device_name(0)
    _stage_status('environment','ready',device=device,torch=torch.__version__)
    setup=pd.DataFrame([
        {'field':'experiment_arm','value':ARM},{'field':'compute_device','value':'cuda'},
        {'field':'cuda_device_name','value':device},{'field':'pool_molecules','value':POOL_N},
        {'field':'validation_molecules','value':VAL_N},{'field':'shadow_models_baseline','value':N_SHADOWS},
        {'field':'epochs_baseline','value':EPOCHS},{'field':'model_layers','value':2},
        {'field':'model_width','value':128},{'field':'heads','value':4},
        {'field':'context_tokens','value':128},{'field':'batch_size','value':32},
        {'field':'seed_scheme','value':'stage-specific fixed seed; shadow k membership seeded by k'},
    ])
    setup.to_csv(ROOT/'results'/'cuda_arm'/'setup.csv',index=False)
    pool,val,meta=_prepare_data(); tok=_tokenizer(pool,val,meta)
    if len(pool)<35000: raise ValueError(f'Expected 35k cached molecules; got {len(pool)}')
    dispatch={'cuda_baseline':lambda:run_cuda_baseline(pool,val,meta,tok),
              'E3':lambda:run_e3(pool,val,tok),'E5a':lambda:run_e5a(pool,val,meta,tok),
              'pretrain_ft':lambda:run_pretrain_ft(pool,val,tok),
              'DP_SGD':lambda:run_dp_sgd(pool,val,tok),'E6':lambda:run_e6(pool,tok)}
    for stage in stages:
        if stage not in dispatch: raise ValueError(f'Unknown stage: {stage}')
        _stage_status(stage,'running')
        try: dispatch[stage]()
        except Exception as exc:
            _stage_status(stage,'error',error=repr(exc))
            raise
